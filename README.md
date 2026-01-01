####MLOps-Term-Project: End-to-End MLOps System###
This project implements a comprehensive machine learning lifecycle for predicting user ad-click behavior. It is designed to meet MLOps Level 2 maturity standards, incorporating automated CI/CD pipelines, experiment tracking, containerization, and continuous monitoring.

#Team and Role Distribution#

*Hasan Can Güler - Data Engineer: Feature Engineering, Hashing, and Feature Cross patterns.
*Ayşe Lara Güneş - ML Engineer (Resilience): Imbalance analysis, Upsampling, and Training Checkpoints.
*Yaren Demirol - ML Engineer: Model Architecture (RF, XGB) and Ensemble Design.
*Gonca Yıldız - MLOps Engineer: MLflow tracking, Experiment logging, and Model Registry.
*Süreyya Yıldırım - DevOps Engineer: CI/CD Pipeline design, GitHub Actions, and Unit Testing.
*Sude Nur Yağ - Deployment Engineer: Dockerization, Stateless FastAPI Serving, and Kubernetes.
*Dolunay Çimen - Monitoring Engineer: Continuous Model Evaluation (CME) and Drift Detection.

#Project Architecture and Folder Structure#
'''
MLOps-Term-Project/
├── data/                       # Data storage & statistical baselines
│   ├── advertising.csv         # Raw dataset
│   └── feature_baseline_stats.csv # Baseline for drift monitoring
├── src/                        # Source code
│   ├── features/               # Hashing, Scaling, and Feature Engineering
│   ├── training/               # Model training scripts (XGBoost, Random Forest)
│   ├── serving/                # FastAPI application (Swagger UI)
│   ├── monitoring/             # Data drift and quality checks
│   ├── evaluation/             # Model performance analysis
│   └── tests/                  # Pytest unit tests
├── pipelines/                  # Training pipeline definitions
├── Dockerfile                  # Containerization setup
├── deployment.yaml             # Kubernetes Deployment manifest
├── service.yaml                # Kubernetes Service manifest
├── requirements.txt            # Python dependencies
└── final_deployment_model.pkl  # Final production-ready Ensemble model



#Installation and Usage#

To run the project locally using Docker and access the Swagger UI:
Clone the Repository:
'''
git clone https://github.com/yarendemirol/MLOps-Term-Project.git
cd MLOps-Term-Project

Build Docker İmage
'''
docker build -t mlops-term-project .

Run the Container:
'''
docker run -p 8000:8000 mlops-term-project

Access Swagger UI: Open your browser and go to:

''' 
http://localhost:8000/docs
