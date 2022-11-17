# Automated-Code-Generation

The replication package is created for the paper titled "On the Reliability and Explainability of Automated Code Generation Approaches (Replicability Study)"

## Getting Started
### Prerequisite and Setup

- Python 3.6 +
- Packages:

```shell
pip install -r requirements.txt
```

Choose a diretory and:

```shell
git clone https://github.com/ReplicateCodeGeneration/Automated-Code-Generation

cd Automated-Code-Generation/
```

### How to reproduce the code generation models

#### ICSE'19 
    ```shell
    cd icses19/
    sh run.sh
    ```
#### TOSEM'19
    ```shell
    cd tosem19/
    sh run.sh
    ```
#### NIPS'21
    ```shell
    cd nips21/
    sh run.sh
    ```
#### FSE'22
Before you start to run experiments with FSE'22, please download the [datasets](https://zenodo.org/record/6900648) first.
    ```shell
    cd fse22/
    sh run.sh
    ```

### File Structure

```shell
.
├── ecco # the code of ECCO for explaination
├── fse22   # the code of FSE'22
|  ├── Code_Refinement
|  ├── outputs
|  ├── run.sh
├── icses19 # the code of ICSE'19
|  ├── data
|  ├── outputs
|  ├── run.sh
├── nips21  # the code of NIPS'21
|  ├── data
|  ├── outputs
|  ├── run.sh
├── tosem19     # the code of TOSEM'19
|  ├── data
|  ├── outputs
|  ├── run.sh
├── data_duplication_analysis.py   # the code for data duplication analysis
├── get_exp.py  # the code for getting the experiment results
├── model_sensitivity_analysis.py   # the code for model sensitivity analysis
