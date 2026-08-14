# Bahraini Food Explorer

A Streamlit object-detection app created for the Bahraini Food Object Detection Lab. The app uses fine-tuned YOLO weights to locate and identify Bahraini foods in an uploaded image or camera photo.

## Food classes

1. Sambosa
2. Kubba
3. Dates
4. Balaleet
5. Luqaimat
6. Nekhy
7. Khanfroosh
8. Halwa
9. Mattai
10. Thareed

The spellings and capitalization in the app should match the class names used when annotating and training the dataset.

## Project files

- `app.py`: Streamlit application
- `requirements.txt`: packages required to run and deploy the application
- `best.pt`: fine-tuned YOLO model weights (add this file after training)

## Add the trained model

After training the YOLO model, copy the generated `best.pt` file into this folder. It must be beside `app.py`:

```text
bahraini_food_streamlit/
├── app.py
├── best.pt
├── README.md
└── requirements.txt
```

## Run locally

Open Command Prompt in the project folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## App features

- Upload JPG, JPEG, or PNG images
- Take a photo using the camera
- Change the minimum detection confidence
- Draw bounding boxes for multiple food objects
- Show the detected class, confidence, and bounding-box coordinates
- Count detections for each food class
- Download the annotated prediction image

## Deploy on Streamlit Community Cloud

1. Add `app.py`, `requirements.txt`, `README.md`, and `best.pt` to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Select **Create app**.
4. Select the GitHub repository and branch.
5. Set the main file path to `app.py`.
6. Select **Deploy**.

If `best.pt` is larger than GitHub's normal file limit, use Git LFS or host the model in a supported model store.

## Lab connection

Use only unseen images for the final demo. Show successful detections as well as some failure cases. The app is the deployment/demo component; dataset analysis, model training, evaluation metrics, error analysis, and the v1-versus-v2 comparison should remain in the project notebooks and report.
