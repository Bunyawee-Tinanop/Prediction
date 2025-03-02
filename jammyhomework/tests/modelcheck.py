import pickle

model_path = "gb_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(type(model))  # ดูว่า model เป็นประเภทอะไร
