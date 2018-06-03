import os

data_path = 'C:\\python\\source\\Portpolio\\brain_tumor\\BRATS\\training'
# for root, dirs, files in os.walk(data_path):
#     print(root,'         ', dirs,'          ', files)
x_path_list = []
y_path_list = []

for data_path in data_path:
    for root, dirs, files in os.walk(data_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # windows에서는 path가 안 읽힘 : \x나 그런 식으로 바꿔야 될듯함.
            if 'MR_Flair' in dir_path:
                if len(os.listdir(dir_path)) != 0:
                    x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]

                    y_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                    # y_path_list = [path.replace('/x/', '/x_filtered/') for path in y_path_list]
                    y_path_list = [path.replace('/x/', '/y/') for path in y_path_list]

                    # images_files = self._sort_by_number(x_path_list)
                    # labels_files = self._sort_by_number(y_path_list)

print(x_path_list)
print(y_path_list)