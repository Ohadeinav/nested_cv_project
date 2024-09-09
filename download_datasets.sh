#! /bin/bash

output_dir=~/Documents/data/nested_cv_project


if [ ! -d $output_dir ]; then
  echo "Output directory doesn't exist: $output_dir"
  echo "Please create it in order to proceed."
  exit 1
fi

curl -LO

cd $output_dir

echo "Done!"