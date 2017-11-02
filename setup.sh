#!/bin/bash

# Copyright 2015-2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mkdir tmp
cd tmp

# Set up diabetes dataset
unzip ../data/dataset_diabetes.zip

# Set up iris dataset
unzip ../data/iris.zip

# Set up wine dataset
unzip ../data/wine.zip

# Set up student dataset
unzip ../data/student.zip

# Set up prostate cancer dataset
unzip ../data/prostate.zip

# Set up car loan dataset
unzip ../data/car_loan.zip

# Set up breast cancer diagnosis dataset
unzip ../data/breast_cancer_diag.zip

# Set up breast cancer prognosis dataset
unzip ../data/breast_cancer_prog.zip

# Set up car value dataset
unzip ../data/car.zip

# Set up dermatology dataset
unzip ../data/dermatology.zip

# Set up fertility dataset
unzip ../data/fertility.zip

# Set up post operative data
unzip ../data/post_operative.zip

# Set up auto mpg data
unzip ../data/auto_mpg.zip
