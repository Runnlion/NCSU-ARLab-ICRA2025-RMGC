import matplotlib.pyplot as plt

# Parsed input data
bbox_data = {
    'AP': 40.95, 'AP50': 70.24, 'AP75': 43.54,
    'AP-1-Cheez-it': 24.36, 'AP-2-Starkist_Tuna': 77.91, 'AP-3-Scissors': 7.52,
    'AP-4-Frenchs_Mustard': 68.42, 'AP-5-Tomato_Soup': 76.84, 'AP-6-Foam_Brick': 61.94,
    'AP-7-Clamp': 23.57, 'AP-8-Plastic_Banana': 48.88, 'AP-9-Mug': 65.10,
    'AP-10-meat_can': 56.26, 'AP-31-Plastic_White_Cup': 55.89, 'AP-32-Wine_Glass': 11.13,
    'AP-33-Key': 18.61, 'AP-34-Nail': 22.07, 'AP-35-Laugh_Out_Loud_Joke_Book': 19.17,
    'AP-36-Adjustable_Wrench': 8.74, 'AP-37-T-shirt': 35.23, 'AP-38-Rolodex_Jumbo_Pencil_Cup': 76.23,
    'AP-39-Glove': 0.17, 'AP-40-Pringles': 60.95
}

segm_data = {
    'AP': 38.20, 'AP50': 59.80, 'AP75': 45.88,
    'AP-1-Cheez-it': 30.96, 'AP-2-Starkist_Tuna': 76.58, 'AP-3-Scissors': 5.68,
    'AP-4-Frenchs_Mustard': 76.34, 'AP-5-Tomato_Soup': 80.69, 'AP-6-Foam_Brick': 74.11,
    'AP-7-Clamp': 10.38, 'AP-8-Plastic_Banana': 40.31, 'AP-9-Mug': 56.24,
    'AP-10-meat_can': 46.14, 'AP-31-Plastic_White_Cup': 64.85, 'AP-32-Wine_Glass': 3.31,
    'AP-33-Key': 9.64, 'AP-34-Nail': 0.0, 'AP-35-Laugh_Out_Loud_Joke_Book': 6.38,
    'AP-36-Adjustable_Wrench': 0.49, 'AP-37-T-shirt': 49.06, 'AP-38-Rolodex_Jumbo_Pencil_Cup': 73.37,
    'AP-39-Glove': 0.13, 'AP-40-Pringles': 59.38
}

# Remove general metrics and focus on per-class performance
general_keys = {'AP', 'AP50', 'AP75'}
bbox_class = {k: v for k, v in bbox_data.items() if k not in general_keys}
segm_class = {k: v for k, v in segm_data.items() if k not in general_keys}


# Filter and sort data for classes 1-10 and 31-40
def get_sorted_data_by_range(data_dict, start, end):
    filtered = {k: v for k, v in data_dict.items() if k.startswith(f'AP-{start}') or k.startswith(f'AP-{end}') or (k.startswith('AP-') and start <= int(k.split('-')[1].split('-')[0]) <= end)}
    sorted_items = sorted(filtered.items(), key=lambda item: item[1])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    return labels, values

# 1-10 classes
labels_1_10, bbox_1_10 = get_sorted_data_by_range(bbox_class, 1, 10)
_, segm_1_10 = get_sorted_data_by_range(segm_class, 1, 10)

# 31-40 classes
labels_31_40, bbox_31_40 = get_sorted_data_by_range(bbox_class, 31, 40)
_, segm_31_40 = get_sorted_data_by_range(segm_class, 31, 40)

# Plot for 1-10
fig1, ax1 = plt.subplots(figsize=(12, 5))
x1 = range(len(labels_1_10))
bar_width = 0.4

ax1.bar(x1, bbox_1_10, width=bar_width, label='BBox AP')
ax1.bar([i + bar_width for i in x1], segm_1_10, width=bar_width, label='Segm AP')

ax1.set_xticks([i + bar_width / 2 for i in x1])
ax1.set_xticklabels(labels_1_10, rotation=90)
ax1.set_ylabel('AP')
ax1.set_title('Per-Class AP: Classes 1-10 (Sorted)')
ax1.legend()
ax1.grid(axis='y')

# Plot for 31-40
fig2, ax2 = plt.subplots(figsize=(12, 5))
x2 = range(len(labels_31_40))

ax2.bar(x2, bbox_31_40, width=bar_width, label='BBox AP')
ax2.bar([i + bar_width for i in x2], segm_31_40, width=bar_width, label='Segm AP')

ax2.set_xticks([i + bar_width / 2 for i in x2])
ax2.set_xticklabels(labels_31_40, rotation=90)
ax2.set_ylabel('AP')
ax2.set_title('Per-Class AP: Classes 31-40 (Sorted)')
ax2.legend()
ax2.grid(axis='y')

plt.tight_layout()
plt.show()
