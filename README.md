# Bahraini Food Explorer

A Streamlit object-detection app created for the Bahraini Food Object Detection Lab. The app uses fine-tuned YOLO weights to locate and identify Bahraini foods in an uploaded image or camera photo.

## Live Demo

Try the deployed app here:

[Open Bahraini Food Explorer](https://bh-food-jtkqkwoe8wulop4yzfnmhi.streamlit.app/)

## Food Classes

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

## Project Files

- `app.py`: Streamlit application
- `requirements.txt`: packages required to run and deploy the application
- `best.pt`: fine-tuned YOLO model weights

## Add the Trained Model

After training the YOLO model, copy the generated `best.pt` file into this folder. It must be beside `app.py`:

```text
bahraini_food_streamlit/
├── app.py
├── best.pt
├── README.md
└── requirements.txt
