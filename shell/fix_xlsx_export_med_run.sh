#!/bin/bash

# MedTable  = MedReceipt   (from entity_extractor) : 진료비 영수증
# MedDetail = MedStatement (from entity_extractor) : 진료비 세부산정내역

# data_folder="MedDetail_good_1"
# data_folder="MedTable_small_1"
data_folder=$1 # use first input argument as data_folder

if [[ -z $1 ]];
then
    echo "Need to pass data_folder as first argument!."
    echo "Exiting..."
    exit 1
else
    echo "Parameter passed = $1"
fi


work_dir="/home/jinho/Projects/lovit/fix_xlsx_export/" ; cd $work_dir
model_path="/home/jinho/models/enhance_xlsx_writer"
dataset="/home/jinho/dataset/MedDataset/$data_folder/images"
output="/home/jinho/dataset/MedDataset/output_$data_folder"

# change data_type accordingly
data_type="MedTable"
if [[ "$data_folder" == "MedDetail" ]]; then
   data_type="MedDetail"
fi


echo "                                                        "
echo "--------------- running fix_xlsx_export --------------- "
echo "                                                        "
echo "data_type    " : $data_type
echo "pythonpath   " : $PYTHONPATH
echo "pip path     " : $(which pip)
echo "python path  " : $(which python)
echo "pwd          " : $(pwd)
echo "work_dir     " : $work_dir
echo "model_path   " : $model_path
echo "dataset      " : $dataset
echo "output       " : $output

# set -x

run_cmd="
python demo/table_exporter/run.py
    --detector-path             $model_path/detection/model.ts
    --detector-metadata         $model_path/detection/metadata.yaml
    --recognizer-path           $model_path/recognition/model.ts
    --recognizer-metadata       $model_path/recognition/metadata.yaml
    --entity-extractor-path     $model_path/entity_extractor/$data_type/model.ts
    --entity-extractor-metadata $model_path/entity_extractor/$data_type/metadata.yaml
    --entity-linker-path        $model_path/entity_linker/$data_type/model.ts
    --entity-linker-metadata    $model_path/entity_linker/$data_type/metadata.yaml
    --testset-dir               $dataset
    --vis
    --xlsx
    --vis-dir                   $output/vis_output
    --xlsx-dir                  $output/xlsx_output
"


echo "                                                        "
echo "--------------- running command         --------------- "
echo "                                                        "
echo -e "$run_cmd"

# $run_cmd
