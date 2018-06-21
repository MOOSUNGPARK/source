# import os
# import numpy as np
# data_path = 'D:\\dataset\\서울대치과병원_Cyst\\cyst\\data\\'
# label_path = 'D:\\dataset\\서울대치과병원_Cyst\\cyst\\labels\\'
#
#
# labels = []
# for root_list, dir_list, file_list in os.walk(label_path):
#     for file in file_list:
#         if '.txt' in file:
#             with open(data_path + '\\' + file) as f:
#                 # a, b, c, d , e = f.read().split(' ')
#                 # print(a)
#                 print(np.loadtxt(f))
#                 labels.append(np.loadtxt(f))
#
# # print(labels)
# # print(labels[:])
import pydicom
import pylab
dFile=pydicom.read_file('d:\\00014_X_029692.dcm') #path to file
# pylab.imshow(dFile.pixel_array,cmap=pylab.cm.bone) # pylab readings and conversion
pylab.imsave('d:\\00014_X_029692_convert.jpg', dFile.pixel_array, cmap=pylab.cm.bone)

# 좌상
# Patient's Name               AN_NM_20170918122303
# Patient's Sex                O
# Patient ID                   AN_ID_20170918122303
# Patient's Birth Date         20170918
# Modality                     MR
# 좌하
# Series Description           3DTOF_200
# Slice Thickness              1.2
# Repetition Time              24
# Echo Time                    3.451
# 우상
# Manufacturer's Model Name    Achieva
# Instance Creation Date       20161122
# Performed Procedure Step Start Time   071412 or Performed Procedure Step End Time
# 우하
# [Window Center]              b'424 '
# [Window Width]               b'737 '

head_list = ['PatientName', 'PatientSex', 'PatientID', 'PatientBirthDate', 'Modality',
             'SeriesDescription', 'SliceThickness', 'RepetitionTime', 'EchoTime',
             'ManufacturerModelName', 'InstanceCreationDate', 'PerformedProcedureStepStartTime',
             'WindowCenter', 'WindowWidth']
value_list = []

import pydicom

dicom_path = 'd:\\FILE00170.dcm'
header = pydicom.read_file(dicom_path, force=True)
print(header.SOPClassUID)
print(header.PatientID)

# h_list = ['SOPClassUID', 'PatientID']
# v_list = ['a', 'b']
# for i, h in enumerate(h_list):
#     exec('header.{0} = v_list[{1}]'.format(h, i))
#
# print(header.SOPClassUID)
# print(header.PatientID)

print(header.WindowWidth)