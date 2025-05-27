import numpy as np
import json
import csv

# Укажи путь до своего npz-файла:
npz_path = "D:\\house_diffusion\\processed_rplan\\rplan_eval_8.npz"

data = np.load(npz_path, allow_pickle=True)

print("Поля (ключи) в npz-файле:")
for key in data.keys():
    print("-", key)
# Имя файлов

filenames = data["filenames"]  # массив имён файлов (по порядку, индекс соответствует houses, graphs и т.д.)
# Примерно: ["datasets/rplan/000001.json", "datasets/rplan/000002.json", ...]

# Если хочешь добавить метаинфу, например, количество комнат:
houses = data["houses"]
num_rooms = [np.sum(house[:,0] != 0) for house in houses]  # примитивная оценка, можно сделать точнее!

min_len = min(len(filenames), len(houses))
whitelist = []
for idx in range(min_len):
    whitelist.append({
        "index": int(idx),
        "filename": str(filenames[idx]),
        "num_rooms": int(num_rooms[idx])
    })

# CSV
with open("whitelist.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "filename", "num_rooms"])
    writer.writeheader()
    writer.writerows(whitelist)

print("Whitelist успешно создан! Смотри whitelist.json и whitelist.csv")
