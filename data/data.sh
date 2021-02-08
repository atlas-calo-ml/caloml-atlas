# Getting our trial data from some fixed locations (for now.)
user=jaofferm

type=$1 #this will indicate the kind of dataset we want to download

if [[ "$type" == "jet" ]]
then
    mkdir jet
    scp -r ${user}@lxplus.cern.ch:/eos/user/j/jaofferm/atlas-calo-ml/data/jet/*/*.root ./jet/
elif [[ "$type" == "pion" ]]
then
    mkdir pion
    scp ${user}@lxplus.cern.ch:/eos/user/m/mswiatlo/images/v10_wJets_wTrack/*.root ./pion/
else
    echo "Type ${type} not understood."
fi
