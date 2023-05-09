CDF   �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 11:46:04 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp85/full_data//NorESM1-ME_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp85/full_data//NorESM1-ME_r1i1p1_ts_185001-210012_globalmean_am_timeseries.nc
Sun Feb 28 11:46:03 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp85/full_data//NorESM1-ME_r1i1p1_ts_185001-210012_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//NorESM1-ME_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc
Sun Feb 28 11:46:01 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//NorESM1-ME_r1i1p1_ts_185001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp85/full_data//NorESM1-ME_r1i1p1_ts_185001-210012_fulldata.nc
Sat May 06 11:39:25 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:39:23 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/NorESM1-ME/r1i1p1/ts_Amon_NorESM1-ME_historical_r1i1p1_185001-200512.nc /echam/folini/cmip5/historical//tmp_01.nc
2012-03-01T07:05:04Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source       +NorESM1-ME 2011  atmosphere: CAM-Oslo (CAM4-Oslo-noresm-ver1_cmip5-r139, f19L26);  ocean: MICOM (MICOM-noresm-ver1_cmip5-r139, gx1v6L53);  ocean biogeochemistry: HAMOCC (HAMOCC-noresm-ver1_cmip5-r139, gx1v6L53);  sea ice: CICE (CICE4-noresm-ver1_cmip5-r139);  land: CLM (CLM4-noresm-ver1_cmip5-r139)    institution       Norwegian Climate Centre   institute_id      NCC    experiment_id         
historical     model_id      
NorESM1-ME     forcing       GHG, SA, Oz, Sl, Vl, BC, OC    parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time                  contact       =Please send any requests or bug reports to noresm-ncc@met.no.      initialization_method               physics_version             tracking_id       $6a6e79fe-bec3-45f0-98cd-9a1ddbeb1a5c   product       output     
experiment        
historical     	frequency         year   creation_date         2012-03-01T07:05:04Z   
project_id        CMIP5      table_id      >Table Amon (01 February 2012) 81f919710c21dca8a1753166d5bac090     title         5NorESM1-ME model output prepared for CMIP5 historical      parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.7.1      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T           |   	time_bnds                             �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           l   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           t   ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         TS     cell_methods      time: mean     history       �2012-03-01T07:05:04Z altered by CMOR: replaced missing value flag (1e+20) with standard missing value (1e+20). 2012-03-01T07:05:04Z altered by CMOR: Converted type from 'd' to 'f'.       associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_NorESM1-ME_historical_r0i0p0.nc areacella: areacella_fx_NorESM1-ME_historical_r0i0p0.nc          �                Aq���   Aq��P   Aq�P   C�w�Aq�6�   Aq�P   Aq��P   C�~ Aq���   Aq��P   Aq��P   C�aRAq��   Aq��P   Aq�dP   C�Q�Aq���   Aq�dP   Aq��P   C�_�Aq���   Aq��P   Aq�FP   C�luAq�k�   Aq�FP   Aq��P   C�Z(Aq���   Aq��P   Aq�(P   C�[�Aq�M�   Aq�(P   Aq��P   C�j+Aq���   Aq��P   Aq�
P   C�{�Aq�/�   Aq�
P   Aq�{P   C�h0Aq���   Aq�{P   Aq��P   C�_�Aq��   Aq��P   Aq�]P   C�i}AqĂ�   Aq�]P   Aq��P   C�ePAq���   Aq��P   Aq�?P   C�Y:Aq�d�   Aq�?P   Aq˰P   C�X1Aq���   Aq˰P   Aq�!P   C�VAq�F�   Aq�!P   AqВP   C�YKAqз�   AqВP   Aq�P   C�^�Aq�(�   Aq�P   Aq�tP   C�k	Aqՙ�   Aq�tP   Aq��P   C�hbAq�
�   Aq��P   Aq�VP   C�p)Aq�{�   Aq�VP   Aq��P   C�k�Aq���   Aq��P   Aq�8P   C�GjAq�]�   Aq�8P   Aq�P   C�WpAq���   Aq�P   Aq�P   C�g�Aq�?�   Aq�P   Aq�P   C�Y�Aq��   Aq�P   Aq��P   C�e�Aq�!�   Aq��P   Aq�mP   C�e'Aq��   Aq�mP   Aq��P   C�MVAq��   Aq��P   Aq�OP   C�c�Aq�t�   Aq�OP   Aq��P   C�b�Aq���   Aq��P   Aq�1P   C�_Aq�V�   Aq�1P   Aq��P   C�`-Aq���   Aq��P   Aq�P   C�/FAq�8�   Aq�P   Aq��P   C�E�Aq���   Aq��P   Aq��P   C�G�Aq��   Aq��P   ArfP   C�R"Ar��   ArfP   Ar�P   C�J�Ar��   Ar�P   ArHP   C�Y�Arm�   ArHP   Ar�P   C�M}Ar��   Ar�P   Ar*P   C�J|ArO�   Ar*P   Ar�P   C�MHAr��   Ar�P   ArP   C�W�Ar1�   ArP   Ar}P   C�UUAr��   Ar}P   Ar�P   C�m~Ar�   Ar�P   Ar_P   C�j�Ar��   Ar_P   Ar�P   C�`�Ar��   Ar�P   ArAP   C�SmArf�   ArAP   Ar�P   C�a�Ar��   Ar�P   Ar!#P   C�_-Ar!H�   Ar!#P   Ar#�P   C�UhAr#��   Ar#�P   Ar&P   C�[Ar&*�   Ar&P   Ar(vP   C�JjAr(��   Ar(vP   Ar*�P   C�?LAr+�   Ar*�P   Ar-XP   C�EnAr-}�   Ar-XP   Ar/�P   C�]yAr/��   Ar/�P   Ar2:P   C�f�Ar2_�   Ar2:P   Ar4�P   C�\�Ar4��   Ar4�P   Ar7P   C�M�Ar7A�   Ar7P   Ar9�P   C�iAr9��   Ar9�P   Ar;�P   C�_�Ar<#�   Ar;�P   Ar>oP   C�^�Ar>��   Ar>oP   Ar@�P   C�[�ArA�   Ar@�P   ArCQP   C�r�ArCv�   ArCQP   ArE�P   C�jArE��   ArE�P   ArH3P   C�\fArHX�   ArH3P   ArJ�P   C�ZArJ��   ArJ�P   ArMP   C�q�ArM:�   ArMP   ArO�P   C�]�ArO��   ArO�P   ArQ�P   C�^�ArR�   ArQ�P   ArThP   C�h�ArT��   ArThP   ArV�P   C�^ArV��   ArV�P   ArYJP   C�X�ArYo�   ArYJP   Ar[�P   C�k�Ar[��   Ar[�P   Ar^,P   C�{�Ar^Q�   Ar^,P   Ar`�P   C�r�Ar`��   Ar`�P   ArcP   C�uArc3�   ArcP   AreP   C�i�Are��   AreP   Arg�P   C�`�Arh�   Arg�P   ArjaP   C�n�Arj��   ArjaP   Arl�P   C�u�Arl��   Arl�P   AroCP   C���Aroh�   AroCP   Arq�P   C�l�Arq��   Arq�P   Art%P   C�voArtJ�   Art%P   Arv�P   C�p�Arv��   Arv�P   AryP   C�rNAry,�   AryP   Ar{xP   C�|�Ar{��   Ar{xP   Ar}�P   C���Ar~�   Ar}�P   Ar�ZP   C��4Ar��   Ar�ZP   Ar��P   C�}�Ar���   Ar��P   Ar�<P   C�v�Ar�a�   Ar�<P   Ar��P   C�pnAr���   Ar��P   Ar�P   C�rAr�C�   Ar�P   Ar��P   C�sAr���   Ar��P   Ar� P   C�~\Ar�%�   Ar� P   Ar�qP   C��@Ar���   Ar�qP   Ar��P   C�l�Ar��   Ar��P   Ar�SP   C�^Ar�x�   Ar�SP   Ar��P   C�Z�Ar���   Ar��P   Ar�5P   C�xVAr�Z�   Ar�5P   Ar��P   C�\�Ar���   Ar��P   Ar�P   C�^~Ar�<�   Ar�P   Ar��P   C�g�Ar���   Ar��P   Ar��P   C�sAr��   Ar��P   Ar�jP   C�z�Ar���   Ar�jP   Ar��P   C�g�Ar� �   Ar��P   Ar�LP   C�r�Ar�q�   Ar�LP   Ar��P   C��Ar���   Ar��P   Ar�.P   C��Ar�S�   Ar�.P   Ar��P   C��CAr���   Ar��P   Ar�P   C��:Ar�5�   Ar�P   Ar��P   C�f-Ar���   Ar��P   Ar��P   C�_�Ar��   Ar��P   Ar�cP   C�[�Ar���   Ar�cP   Ar��P   C�eAr���   Ar��P   Ar�EP   C�jVAr�j�   Ar�EP   ArĶP   C�v�Ar���   ArĶP   Ar�'P   C�pBAr�L�   Ar�'P   ArɘP   C�[zArɽ�   ArɘP   Ar�	P   C�d9Ar�.�   Ar�	P   Ar�zP   C�q{ArΟ�   Ar�zP   Ar��P   C�w�Ar��   Ar��P   Ar�\P   C�v�ArӁ�   Ar�\P   Ar��P   C�l�Ar���   Ar��P   Ar�>P   C�a�Ar�c�   Ar�>P   ArگP   C�n�Ar���   ArگP   Ar� P   C�r�Ar�E�   Ar� P   ArߑP   C���Ar߶�   ArߑP   Ar�P   C��}Ar�'�   Ar�P   Ar�sP   C��!Ar��   Ar�sP   Ar��P   C���Ar�	�   Ar��P   Ar�UP   C���Ar�z�   Ar�UP   Ar��P   C�~Ar���   Ar��P   Ar�7P   C��rAr�\�   Ar�7P   Ar�P   C��
Ar���   Ar�P   Ar�P   C�~�Ar�>�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C���Ar� �   Ar��P   Ar�lP   C��Ar���   Ar�lP   Ar��P   C���Ar��   Ar��P   Ar�NP   C���Ar�s�   Ar�NP   As�P   C�rEAs��   As�P   As0P   C�n�AsU�   As0P   As�P   C�|�As��   As�P   As	P   C���As	7�   As	P   As�P   C���As��   As�P   As�P   C��}As�   As�P   AseP   C��eAs��   AseP   As�P   C���As��   As�P   AsGP   C���Asl�   AsGP   As�P   C��|As��   As�P   As)P   C���AsN�   As)P   As�P   C���As��   As�P   AsP   C�ԛAs0�   AsP   As!|P   C�ԃAs!��   As!|P   As#�P   C��xAs$�   As#�P   As&^P   C���As&��   As&^P   As(�P   C��As(��   As(�P   As+@P   C��rAs+e�   As+@P   As-�P   C��As-��   As-�P   As0"P   C��DAs0G�   As0"P   As2�P   C���As2��   As2�P   As5P   C��As5)�   As5P   As7uP   C�ϯAs7��   As7uP   As9�P   C���As:�   As9�P   As<WP   C���As<|�   As<WP   As>�P   C���As>��   As>�P   AsA9P   C���AsA^�   AsA9P   AsC�P   C��tAsC��   AsC�P   AsFP   C���AsF@�   AsFP   AsH�P   C��-AsH��   AsH�P   AsJ�P   C��cAsK"�   AsJ�P   AsMnP   C��AsM��   AsMnP   AsO�P   C��AsP�   AsO�P   AsRPP   C� �AsRu�   AsRPP   AsT�P   C���AsT��   AsT�P   AsW2P   C���AsWW�   AsW2P   AsY�P   C�7AsY��   AsY�P   As\P   C��As\9�   As\P   As^�P   C��As^��   As^�P   As`�P   C��Asa�   As`�P   AscgP   C��Asc��   AscgP   Ase�P   C�$�Ase��   Ase�P   AshIP   C�%�Ashn�   AshIP   Asj�P   C�)�Asj��   Asj�P   Asm+P   C�FAsmP�   Asm+P   Aso�P   C�!�Aso��   Aso�P   AsrP   C�&YAsr2�   AsrP   Ast~P   C�!�Ast��   Ast~P   Asv�P   C�/�Asw�   Asv�P   Asy`P   C�,�Asy��   Asy`P   As{�P   C�;�As{��   As{�P   As~BP   C�G�As~g�   As~BP   As��P   C�MgAs���   As��P   As�$P   C�I�As�I�   As�$P   As��P   C�IDAs���   As��P   As�P   C�S�As�+�   As�P   As�wP   C�^9As���   As�wP   As��P   C�a.As��   As��P   As�YP   C�cAs�~�   As�YP   As��P   C�l�As���   As��P   As�;P   C�l�As�`�   As�;P   As��P   C�z]As���   As��P   As�P   C�siAs�B�   As�P   As��P   C�q�As���   As��P   As��P   C�x�As�$�   As��P   As�pP   C���As���   As�pP   As��P   C��As��   As��P   As�RP   C��As�w�   As�RP   As��P   C��As���   As��P   As�4P   C��\As�Y�   As�4P   As��P   C��As���   As��P   As�P   C��8As�;�   As�P   As��P   C���As���   As��P   As��P   C��NAs��   As��P   As�iP   C���As���   As�iP   As��P   C���As���   As��P   As�KP   C�˪As�p�   As�KP   As��P   C�ܯAs���   As��P   As�-P   C�ѻAs�R�   As�-P   AsP   C��As���   AsP   As�P   C��<As�4�   As�P   AsǀP   C���Asǥ�   AsǀP   As��P   C��As��   As��P   As�bP   C��)Aṡ�   As�bP   As��P   C��As���   As��P   As�DP   C��As�i�   As�DP   AsӵP   C���As���   AsӵP   As�&P   C�1As�K�   As�&P   AsؗP   C��Asؼ�   AsؗP   As�P   C�-As�-�   As�P   As�yP   C��Asݞ�   As�yP   As��P   C�,As��   As��P   As�[P   C�:uAs��   As�[P   As��P   C�@GAs���   As��P   As�=P   C�)�As�b�   As�=P   As�P   C��As���   As�P   As�P   C�-wAs�D�   As�P   As�P   C�=�As��   As�P   As�P   C�>As�&�   As�P   As�rP   C�MAs��   As�rP   As��P   C�P�As��   As��P   As�TP   C�[?As�y�   As�TP   As��P   C�`�As���   As��P   As�6P   C�YAs�[�   As�6P   As��P   C�WuAs���   As��P   AtP   C�a5At=�   AtP   At�P   C�l�At��   At�P   At�P   C�taAt�   At�P   At	kP   C�xi