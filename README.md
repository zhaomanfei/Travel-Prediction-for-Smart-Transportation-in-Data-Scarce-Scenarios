# Travel-Prediction-for-Smart-Transportation-in-Data-Scarce-Scenarios
This repository contains the source code for travel prediction under data-scarce smart transportation scenarios.

## Project Overview
To tackle unreliable prediction accuracy in data-scarce scenarios, this project uses the real-world California PEMS dataset as the source domain. It leverages SUMO simulation platform to construct the road network of the target city, generates traffic flow data, and builds a hierarchical transfer-learning framework for traffic prediction.

## Environment Requirements
- Python >= 3.8
- TensorFlow / Keras
- SUMO
- numpy
- pandas
- scikit-learn

## File Description
| File Name | Description |
| ---- | ---- |
| `LSTM_Model.py` | LSTM prediction model definition |
| `dataLoad.py` | Data loading module |
| `dataPreat.py` | Raw data preprocessing pipeline |
| `bigData_npz_sumo.py` | SUMO simulation and traffic data generation |
| `control_group_experiment.py` | Baseline / control group comparison experiments |
| `modelTrain_afterCleaning.py` | Main model training script |
| `extract_traffic_features.py` | Traffic spatio-temporal feature extraction |
| `generalization_ability.py` | Model generalization ability test |
| `generate_od.py` | OD matrix generation for simulation |

## How to Reproduce
1. Install SUMO simulation environment and required Python packages
2. Download California PEMS source dataset
3. Run `dataPreat.py` for raw data preprocessing
4. Run `modelTrain_afterCleaning.py` to train the transfer learning model

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
project introduction and reproduction guide
