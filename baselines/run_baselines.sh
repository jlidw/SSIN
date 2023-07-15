# HK dataset
python baseline_idw_ok.py --dataset=hk --method=kriging --variogram=spherical --start_year=2012 --end_year=2014 ;
python baseline_idw_ok.py --dataset=hk --method=idw --power_p=2 --start_year=2012 --end_year=2014 ;
python baseline_tin_tps.py --dataset=hk --method=spline --start_year=2012 --end_year=2014 ;
python baseline_tin_tps.py --dataset=hk --method=tin --start_year=2012 --end_year=2014 ;

# BW dataset
python baseline_idw_ok.py --dataset=bw --method=kriging --variogram=spherical --start_year=2012 --end_year=2014 ;
python baseline_idw_ok.py --dataset=bw --method=idw --power_p=2 --start_year=2012 --end_year=2014 ;
python baseline_tin_tps.py --dataset=bw --method=spline --start_year=2012 --end_year=2014 ;
python baseline_tin_tps.py --dataset=bw --method=tin --start_year=2012 --end_year=2014 ;

