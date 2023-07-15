"""
Modified based on Longformer.
@article{Beltagy2020Longformer,
  title={Longformer: The Long-Document Transformer},
  author={Iz Beltagy and Matthew E. Peters and Arman Cohan},
  journal={arXiv:2004.05150},
  year={2020},
}
"""

from typing import Union
from functools import lru_cache

import torch
import os.path
import sys
# sys.path.append('spaformer/tvm/python')


class GraphTwoTMM(torch.autograd.Function):
    '''Class to encapsulate tvm code for compiling a diagonal_mm function, in addition to calling
    this function from PyTorch
    '''

    function_dict = {}  # save a list of functions, each has a different set of parameters

    @staticmethod
    def _compile_function_two_tensors(dtype: str, device: str, b0: int = 4, b1: int = 8, b2: int = 8):
        '''Compiles a tvm function that computes diagonal_mm
        args:
        dtype: str in ['float64', 'float32', 'float16']
        device: str in ['cpu' or 'cuda']
        b0, b1, b2: size of tensor tiles. Very important for good performance
        '''
        import tvm  # import the full tvm library here for compilation. Don't import at the top of the file in case we don't need to compile
        from tvm.contrib import nvcc
        @tvm.register_func
        def tvm_callback_cuda_compile(code):
            """Use nvcc compiler for better perf."""
            ptx = nvcc.compile_cuda(code, target="ptx", arch='sm_52')  # use old arch for this to work on old GPUs
            return ptx

        assert dtype in ['float16', 'float32', 'float64']
        assert device in ['cpu', 'cuda']
        device = None if device == 'cpu' else device
        tgt_host = "llvm"

        b = tvm.te.var('b')  # batch size
        n = tvm.te.var('n')  # sequence length
        h = tvm.te.var('h')  # number of heads
        m = tvm.te.var('m')  # hidden dimension
        padding = tvm.te.var('padding')  # padding value

        is_t1_diagonaled = tvm.te.var('is_t1_diagonaled')  # denote is_t1_diagonaled
        transpose_t1 = tvm.te.var('transpose_t1')  # t1 should be transposed
        t1d3 = tvm.te.var('t1d3')  # last dimension of t1
        t3d3 = tvm.te.var('t3d3')  # last dimension of t3 (the result tensor)
        X = tvm.te.placeholder((b, n, h, t1d3), name='X', dtype=dtype)  # first tensor
        Y = tvm.te.placeholder((b, n, h, m), name='Y', dtype=dtype)  # second tensor
        k = tvm.te.reduce_axis((0, t1d3), name='k')  # dimension to sum over

        # q_k, k_q: n*n matrix to support any forms of attention calculations (including directed graph);
        # q_k_mask[i,j] = 0 means q_i can NOT get info from k_j; q_k_mask[i,j] = 1 means q_i can get info from k_j.
        # k_q_mask[i,j] = 0 means k_i is NOT attended to q_j;    k_q_mask[i,j] = 1 means k_i is attended to q_j.
        # Different sequence may have different neighbor relationships, so here q_k_mask and k_q_mask have the 'b' dim
        q_k_mask = tvm.te.placeholder((b, n, n), name='q_k', dtype='int')  # index matrix from q to k
        k_q_mask = tvm.te.placeholder((b, n, n), name='k_q', dtype='int')  # index matrix from k to q
        output_shape = (b, n, h, t3d3)  # shape of the result tensor

        algorithm = lambda l, i, q, j: tvm.te.sum(  # can handle two cases of calculations: attn = Q*K; context = attn*V
            tvm.te.if_then_else(
                is_t1_diagonaled == 1,  # i.e., is_t1_diagonaled == True
                tvm.te.if_then_else(
                    transpose_t1 == 0,
                    tvm.te.if_then_else(  # backward: grad_Q = grad_attn * K; forward: context = attn * V
                        q_k_mask[l, i, k] > 0,
                        X[l, i, q, k] * Y[l, k, q, j],
                        padding
                    ),
                    tvm.te.if_then_else(  # backward: grad_K = grad_attn * Q; backward: grad_V =  attn * grad_context
                        k_q_mask[l, i, k] > 0,
                        X[l, k, q, i] * Y[l, k, q, j],
                        padding
                    ),
                ),
                tvm.te.if_then_else(
                    q_k_mask[l, i, j] > 0,  # forward: attn = Q*K; backward: grad_attn =  grad_context * V
                    X[l, i, q, k] * Y[l, j, q, k],
                    padding
                )
            ), axis=k)

        Z = tvm.te.compute(output_shape, algorithm, name='Z')  # automatically generate cuda code
        s = tvm.te.create_schedule(Z.op)

        print('Lowering: \n ===================== \n{}'.format(tvm.lower(s, [X, Y, q_k_mask, k_q_mask], simple_mode=True)))

        # split long axis into smaller chunks and assing each one to a separate GPU thread/block
        ko, ki = s[Z].split(Z.op.reduce_axis[0], factor=b0)
        ZF = s.rfactor(Z, ki)

        j_outer, j_inner = s[Z].split(s[Z].op.axis[-1], factor=b1)
        i_outer, i_inner = s[Z].split(s[Z].op.axis[1], factor=b2)

        s[Z].bind(j_outer, tvm.te.thread_axis("blockIdx.x"))
        s[Z].bind(j_inner, tvm.te.thread_axis("threadIdx.y"))

        s[Z].bind(i_outer, tvm.te.thread_axis("blockIdx.y"))
        s[Z].bind(i_inner, tvm.te.thread_axis("threadIdx.z"))

        tx = tvm.te.thread_axis("threadIdx.x")
        s[Z].bind(s[Z].op.reduce_axis[0], tx)
        s[ZF].compute_at(s[Z], s[Z].op.reduce_axis[0])
        s[Z].set_store_predicate(tx.var.equal(0))

        print('Lowering with GPU splits: \n ===================== \n{}'.format(tvm.lower(s, [X, Y, q_k_mask, k_q_mask], simple_mode=True)))

        # compiling the automatically generated cuda code
        graph_mm_two_vectors = tvm.build(s, [X, Y, Z, q_k_mask, k_q_mask, padding, is_t1_diagonaled, transpose_t1, t3d3],
                                         target=device, target_host=tgt_host, name='graph_mm_two_vectors')
        # print(graph_mm_two_vectors.imported_modules[0].get_source())
        return graph_mm_two_vectors

    @staticmethod
    def _get_lib_filename(dtype: str, device: str):
        base_filename = 'lib-2/lib_two_tensor_mm'
        return '{}_{}_{}.so'.format(base_filename, dtype, device)

    @staticmethod
    def _save_compiled_function(f, dtype: str, device: str):
        if not os.path.exists('lib-2/'):
            os.makedirs('lib-2/')
        f.export_library(GraphTwoTMM._get_lib_filename(dtype, device))

    @staticmethod
    def _load_compiled_function(dtype: str, device: str):
        # from tvm.module import load  # this can be the small runtime python library, and doesn't need to be the whole thing
        from tvm.runtime.module import load_module as load

        filename = GraphTwoTMM._get_lib_filename(dtype, device)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_dirs = ['../../', '../', './', f'{current_dir}/', f'{current_dir}/../']
        for potential_dir in potential_dirs:
            filepath = '{}{}'.format(potential_dir, filename)
            if os.path.isfile(filepath):
                print('Loading tvm binary from: {}'.format(filepath))
                return load(filepath)
        return None

    @staticmethod
    def _get_function(dtype: str, device: str):
        '''Loads the function from the disk or compile it'''
        # A list of arguments that define the function
        args = (dtype, device)
        if args not in GraphTwoTMM.function_dict:
            graph_mm = GraphTwoTMM._load_compiled_function(dtype, device)  # try to load from disk
            if not graph_mm:
                print('Tvm binary not found. Compiling ...')
                graph_mm = GraphTwoTMM._compile_function_two_tensors(dtype, device)  # compile
                GraphTwoTMM._save_compiled_function(graph_mm, dtype, device)  # save to disk
            # convert the tvm function into a pytorch function
            from tvm.contrib import dlpack
            graph_mm_pytorch = dlpack.to_pytorch_func(graph_mm)  # wrap it as a pytorch function
            # save the function into a dictionary to be reused
            GraphTwoTMM.function_dict[args] = graph_mm_pytorch  # save it in a dictionary for next time
        return GraphTwoTMM.function_dict[args]

    @staticmethod
    def _graph_mm(t1: torch.Tensor, t2: torch.Tensor, q_k_mask: torch.Tensor, k_q_mask: torch.Tensor,
                  is_t1_diagonaled: bool = False, transpose_t1: bool = False, padding: int = 0):
        '''Calls the compiled function after checking the input format. This function is called in three different modes.
        t1 x t2 = r ==> t1 and t2 are not diagonaled, but r is. Useful for query x key = attention_scores
        t1 x t2 = r ==> t1 is diagonaled, but t2 and r are not. Useful to compuate attantion_scores x value = context
        t1 x t2 = r ==> t1 is diagonaled and it should be transposed, but t2 and r are not diagonaled. Useful in some of
                            the calculations in the backward pass.
        '''
        dtype = str(t1.dtype).split('.')[1]
        device = t1.device.type
        assert len(t1.shape) == 4
        assert len(t1.shape) == len(t2.shape)
        assert t1.shape[:3] == t2.shape[:3]

        b = t1.shape[0]  # batch size
        n = t1.shape[1]  # sequence length
        h = t1.shape[2]  # number of heads
        m = t2.shape[3]  # hidden dimension
        if is_t1_diagonaled:
            assert t1.shape[3] == n
            r = t1.new_empty(b, n, h, m)  # allocate space for the result tensor
            t3d3 = m
        else:
            assert not transpose_t1
            assert t1.shape[3] == m
            r = t1.new_empty(b, n, h, n)  # allocate space for the result tensor
            t3d3 = n

        # gets function from memory, from disk or compiles it from scratch
        _graph_mm_function = GraphTwoTMM._get_function(dtype=dtype, device=device)

        # The last argument to this function is a little hacky. It is the size of the last dimension of the result tensor
        # We use it as a proxy to tell if t1_is_diagonaled or not (if t1 is diagonaled, result is not, and vice versa).
        # The second reason is that the lambda expression in `_compile_function` is easier to express when the shape of the output is known
        # This functions computes diagonal_mm then saves the result in `r`
        # if m == n:
        #     # FIXME
        #     print('Error: the hidden dimension {m} shouldn\'t match sequence length {n}')
        #     assert False

        # Directly use 'is_t1_diagonaled' to select different cases for calculation
        _graph_mm_function(t1, t2, r, q_k_mask, k_q_mask, padding, is_t1_diagonaled, transpose_t1, t3d3)
        return r

    @staticmethod
    def _prepare_tensors(t):
        '''Fix `stride()` information of input tensor. This addresses some inconsistency in stride information in PyTorch.
        For a tensor t, if t.size(0) == 1, then the value of t.stride()[0] doesn't matter.
        TVM expects this value to be the `product(t.size()[1:])` but PyTorch some times sets it to `t.stride()[1]`.
        Here's an example to reporduce this issue:
            import torch
            print(torch.randn(1, 10).stride())
            > (10, 1)
            print(torch.randn(10, 1).t().contiguous().stride())
            > (1, 1)  # expected it to be (10, 1) as above
            print(torch.randn(10, 2).t().contiguous().stride())
            > (10, 1) # but gets the expected stride if the first dimension is > 1
        '''
        assert t.is_contiguous()
        t_stride = list(t.stride())
        t_size = list(t.size())
        # Fix wrong stride information for the first dimension. This occures when batch_size=1
        if t_size[0] == 1 and t_stride[0] == t_stride[1]:
            # In this case, the stride of the first dimension should be the product
            # of the sizes  of all other dimensions
            t_stride[0] = t_size[1] * t_size[2] * t_size[3]
            t = t.as_strided(size=t_size, stride=t_stride)
        return t

    min_seq_len = 16  # unexpected output if seq_len < 16

    @staticmethod
    def forward(ctx, t1: torch.Tensor, t2: torch.Tensor, q_k_mask, k_q_mask, is_t1_diagonaled: bool = False, padding: int = 0) -> torch.Tensor:
        '''Compuates diagonal_mm of t1 and t2.
        args:
        ctx: ctx is a context object that can be used to stash information for backward computation
        t1: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size|number_of_diagonals).
            t1 can be a regular tensor (e.g. `query_layer`) or a diagonaled one (e.g. `attention_scores`)
        t2: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size). This is always a non-diagonaled
            tensor, e.g. `key_layer` or `value_layer`
        t3:
        is_t1_diagonaled: is t1 a diagonaled or a regular tensor
        padding: the padding value to use when accessing invalid locations. This is mainly useful when the padding
            needs to be a very large negative value (to compute softmax of attentions). For other usecases,
            please use zero padding.
        returns: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size|number_of_diagonals)
            if t1 is diagonaed, result is non-diagonaled, and vice versa
        '''
        seq_len = t1.size(1)
        assert seq_len >= GraphTwoTMM.min_seq_len, 'avoid splitting errors by using seq_len >= {}'.format(GraphTwoTMM.min_seq_len)  # FIXME

        t1 = GraphTwoTMM._prepare_tensors(t1)
        t2 = GraphTwoTMM._prepare_tensors(t2)
        q_k_mask = GraphTwoTMM._prepare_tensors(q_k_mask)
        k_q_mask = GraphTwoTMM._prepare_tensors(k_q_mask)
        ctx.save_for_backward(t1, t2, q_k_mask, k_q_mask)  # save tensors to be used in the backward pass
        ctx.is_t1_diagonaled = is_t1_diagonaled
        output = GraphTwoTMM._graph_mm(t1, t2, q_k_mask, k_q_mask, is_t1_diagonaled=is_t1_diagonaled, padding=padding)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        t1, t2, q_k_mask, k_q_mask = ctx.saved_tensors  # get the saved tensors
        is_t1_diagonaled = ctx.is_t1_diagonaled
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()  # tvm requires all input tensors to be contiguous
        grad_output = GraphTwoTMM._prepare_tensors(grad_output)
        # http://cs231n.github.io/optimization-2/
        # https://pytorch.org/docs/master/notes/extending.html
        grad_t1 = GraphTwoTMM._graph_mm(grad_output, t2, q_k_mask, k_q_mask, is_t1_diagonaled=not is_t1_diagonaled)
        if is_t1_diagonaled:  # attn*V, grad_t2 = grad_V
            grad_t2 = GraphTwoTMM._graph_mm(t1, grad_output, q_k_mask, k_q_mask, is_t1_diagonaled=True, transpose_t1=True)
        else:  # Q*K, grad_t2 = grad_K
            grad_t2 = GraphTwoTMM._graph_mm(grad_output, t1, q_k_mask, k_q_mask, is_t1_diagonaled=True, transpose_t1=True)
        return grad_t1, grad_t2, None, None, None, None, None

graph_two_vec_mm = GraphTwoTMM.apply

