#!/bin/bash 

echo "PlaceHasFeature" 
PlaceHasFeature_triggers="takeout,casual,waiter"

./scripts/make_subsamples_no_source.sh PlaceHasFeature /srv/local1/estengel//resources/data/smcalflow.agent.data /srv/local1/estengel//resources/data/smcalflow_samples_no_source ${PlaceHasFeature_triggers}

# already done 
#FindManager_triggers="boss,manager,supervisor"
#./scripts/make_subsamples_no_source.sh Find/srv/local1/estengel//resources/data/smcalflow.agent.data /srv/local1/estengel//resources/data/smcalflow_samples_no_source ${PlaceHasFeature_triggers}

echo "Tomorrow" 
Tomorrow_triggers="tomorrow"
./scripts/make_subsamples_no_source.sh Tomorrow /srv/local1/estengel//resources/data/smcalflow.agent.data /srv/local1/estengel//resources/data/smcalflow_samples_no_source ${Tomorrow_triggers}

echo "FenceAttendee" 
FenceAttendee_triggers="meet,mom"
./scripts/make_subsamples_no_source.sh FenceAttendee /srv/local1/estengel//resources/data/smcalflow.agent.data /srv/local1/estengel//resources/data/smcalflow_samples_no_source ${FenceAttendee_triggers}

echo "DoNotConfirm" 
DoNotConfirm_triggers="cancel,n't,no"
./scripts/make_subsamples_no_source.sh DoNotConfirm /srv/local1/estengel//resources/data/smcalflow.agent.data /srv/local1/estengel//resources/data/smcalflow_samples_no_source ${DoNotConfirm_triggers}

