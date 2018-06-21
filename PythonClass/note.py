import mritopng

#
# # mritopng.convert_file('d:\\00014_X_029692.dcm', 'd:\\00014_X_029692.png')
# mritopng.mri_to_png('d:\\00014_X_029692.dcm', 'd:\\00014_X_029692.png')
# # pydicom.dcmwrite('00014_X_029692.jpg')



import pydicom
import pylab
dFile=pydicom.read_file('d:\\00014_X_029692.dcm') #path to file
# pylab.imshow(dFile.pixel_array,cmap=pylab.cm.bone) # pylab readings and conversion
pylab.imsave('d:\\00014_X_029692_convert.jpg', dFile.pixel_array, cmap=pylab.cm.bone)
# pylab.show() #Dispaly

