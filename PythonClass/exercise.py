# print(60*60*3 + 30 * 60)
#
# import numpy as np
#
# a= np.array([1.232412351, 12.125152125])
# print(type(a))
#
# b = np.around(a, decimals=2, out=None)
# print(b*100000)
#
#
# import math
# def cosine_similarity(v1,v2):
#     "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
#     sumxx, sumxy, sumyy = 0, 0, 0
#     for i in range(len(v1)):
#         x = v1[i]; y = v2[i]
#         sumxx += x*x
#         sumyy += y*y
#         sumxy += x*y
#     return sumxy/math.sqrt(sumxx*sumyy)
#
# v1,v2 = [3, 45, 7, 2], [2, 54, 13, 15]
# print(v1, v2, cosine_similarity(v1,v2))

# for i in range(12):
#     print(i, round((240/11)*i))
from librosa import time_to_samples
import numpy as np
# time_to_samples(np.arrange(0.1), sr=22050)
# librosa.time_to_samples(np.arange(0, 1, 0.1), sr=22050)

# print(( 240/414) * 51)
# print(( 240/414) * 76)
# print(( 240/414) * 103)
# print(( 240/414) * 182)
# print(( 240/414) * 207)
# print(( 240/414) * 236)
# print(( 240/414) * 312)
# print(( 240/414) * 338)
'''
트와이스 녹녹(0.5)
((42, 41), (61, 60)), ((43, 42), (62, 61)), , ((50, 24), (69, 43)), ((50, 25), (69, 44)), 
((76, 51), (95, 70)), ((77, 52), (96, 71)), ((78, 53), (97, 72)), ((79, 54), (98, 73)), 
((90, 89), (109, 108)), ((91, 90), (110, 109)), ((92, 91), (111, 110)), ((93, 92), (112, 111)), ((94, 93), (113, 112)), ((95, 94), (114, 113)), ((96, 95), (115, 114)), ((97, 96), (116, 115)), ((98, 97), (117, 116)), ((99, 98), (118, 117)), ((100, 99), (119, 118)), ((101, 100), (120, 119)), ((102, 51), (121, 70)), ((102, 76), (121, 95)), ((102, 101), (121, 120)), ((103, 51), (122, 70)), ((103, 52), (122, 71)), ((103, 77), (122, 96)), ((103, 102), (122, 121)), ((104, 52), (123, 71)), ((104, 53), (123, 72)), ((104, 78), (123, 97)), ((104, 103), (123, 122)), ((105, 53), (124, 72)), ((105, 54), (124, 73)), ((105, 79), (124, 98)), ((105, 104), (124, 123)), ((106, 80), (125, 99)), ((106, 105), (125, 124)), ((107, 81), (126, 100)), ((107, 106), (126, 125)), ((108, 107), (127, 126)), 
((129, 128), (148, 147)), 
((166, 86), (185, 105)), ((167, 87), (186, 106)), ((168, 88), (187, 107)), ((169, 89), (188, 108)), ((170, 90), (189, 109)), ((171, 91), (190, 110)), ((172, 92), (191, 111)), ((173, 93), (192, 112)), ((174, 94), (193, 113)), 
((181, 76), (200, 95)), ((181, 102), (200, 121)), ((182, 77), (201, 96)), ((182, 103), (201, 122)), ((183, 78), (202, 97)), ((183, 104), (202, 123)), ((184, 79), (203, 98)), ((184, 105), (203, 124)), ((185, 80), (204, 99)), ((185, 106), (204, 125)), ((186, 81), (205, 100)), ((186, 107), (205, 126)), ((187, 82), (206, 101)), ((188, 83), (207, 102)), ((189, 84), (208, 103)), ((190, 85), (209, 104)), ((191, 86), (210, 105)), ((191, 166), (210, 185)), ((192, 87), (211, 106)), ((192, 167), (211, 186)), ((193, 88), (212, 107)), ((193, 168), (212, 187)), ((194, 89), (213, 108)), ((194, 90), (213, 109)), ((194, 169), (213, 188)), ((195, 90), (214, 109)), ((195, 91), (214, 110)), ((195, 170), (214, 189)), ((196, 91), (215, 110)), ((196, 92), (215, 111)), ((196, 171), (215, 190)), ((197, 92), (216, 111)), ((197, 93), (216, 112)), ((197, 172), (216, 191)), ((198, 93), (217, 112)), ((198, 94), (217, 113)), ((198, 173), (217, 192)), ((199, 94), (218, 113)), ((199, 95), (218, 114)), ((199, 174), (218, 193)), ((200, 95), (219, 114)), ((200, 96), (219, 115)), ((200, 175), (219, 194)), ((201, 96), (220, 115)), ((201, 97), (220, 116)), ((201, 176), (220, 195)), ((202, 97), (221, 116)), ((202, 98), (221, 117)), ((202, 177), (221, 196)), ((202, 201), (221, 220)), ((203, 98), (222, 117)), ((203, 99), (222, 118)), ((203, 178), (222, 197)), ((203, 202), (222, 221)), ((204, 99), (223, 118)), ((204, 100), (223, 119)), ((204, 179), (223, 198)), ((204, 203), (223, 222)), ((205, 100), (224, 119)), ((205, 101), (224, 120)), ((205, 180), (224, 199)), ((205, 204), (224, 223)), ((206, 76), (225, 95)), ((206, 101), (225, 120)), ((206, 102), (225, 121)), ((206, 181), (225, 200)), ((206, 205), (225, 224)), ((207, 51), (226, 70)), ((207, 76), (226, 95)), ((207, 77), (226, 96)), ((207, 102), (226, 121)), ((207, 103), (226, 122)), ((207, 181), (226, 200)), ((207, 182), (226, 201)), ((208, 52), (227, 71)), ((208, 77), (227, 96)), ((208, 78), (227, 97)), ((208, 103), (227, 122)), ((208, 104), (227, 123)), ((208, 182), (227, 201)), ((208, 183), (227, 202)), ((209, 53), (228, 72)), ((209, 78), (228, 97)), ((209, 79), (228, 98)), ((209, 104), (228, 123)), ((209, 105), (228, 124)), ((209, 183), (228, 202)), ((209, 184), (228, 203)), ((210, 54), (229, 73)), ((210, 79), (229, 98)), ((210, 80), (229, 99)), ((210, 105), (229, 124)), ((210, 106), (229, 125)), ((210, 184), (229, 203)), ((210, 185), (229, 204)), ((211, 80), (230, 99)), ((211, 81), (230, 100)), ((211, 106), (230, 125)), ((211, 107), (230, 126)), ((211, 185), (230, 204)), ((211, 186), (230, 205)), ((212, 81), (231, 100)), ((212, 82), (231, 101)), ((212, 107), (231, 126)), ((212, 108), (231, 127)), ((212, 186), (231, 205)), ((212, 187), (231, 206)), ((213, 108), (232, 127)), ((214, 109), (233, 128)), ((215, 110), (234, 129)), ((216, 111), (235, 130)), ((217, 112), (236, 131)), ((218, 113), (237, 132)), ((219, 114), (238, 133)), 
((234, 49), (253, 68)), ((234, 50), (253, 69)), ((235, 50), (254, 69)), ((235, 51), (254, 70)), ((235, 76), (254, 95)), ((235, 102), (254, 121)), ((235, 181), (254, 200)), ((235, 206), (254, 225)), ((235, 207), (254, 226)), ((236, 51), (255, 70)), ((236, 52), (255, 71)), ((236, 77), (255, 96)), ((236, 103), (255, 122)), ((236, 182), (255, 201)), ((236, 207), (255, 226)), ((236, 208), (255, 227)), ((237, 52), (256, 71)), ((237, 53), (256, 72)), ((237, 78), (256, 97)), ((237, 104), (256, 123)), ((237, 183), (256, 202)), ((237, 208), (256, 227)), ((237, 209), (256, 228)), ((238, 53), (257, 72)), ((238, 54), (257, 73)), ((238, 79), (257, 98)), ((238, 105), (257, 124)), ((238, 184), (257, 203)), ((238, 209), (257, 228)), ((238, 210), (257, 229)), ((239, 54), (258, 73)), ((239, 80), (258, 99)), ((239, 106), (258, 125)), ((239, 185), (258, 204)), ((239, 210), (258, 229)), ((239, 211), (258, 230)), ((240, 81), (259, 100)), ((240, 107), (259, 126)), ((240, 186), (259, 205)), ((240, 211), (259, 230)), ((240, 212), (259, 231)), ((241, 82), (260, 101)), ((241, 108), (260, 127)), ((241, 187), (260, 206)), ((241, 212), (260, 231)), ((241, 213), (260, 232)), ((242, 83), (261, 102)), ((242, 188), (261, 207)), ((243, 84), (262, 103)), ((243, 189), (262, 208)), ((244, 85), (263, 104)), ((244, 190), (263, 209)), ((245, 86), (264, 105)), ((245, 166), (264, 185)), ((245, 191), (264, 210)), ((246, 87), (265, 106)), ((246, 167), (265, 186)), ((246, 192), (265, 211)), ((247, 88), (266, 107)), ((247, 168), (266, 187)), ((247, 193), (266, 212)), ((248, 89), (267, 108)), ((248, 169), (267, 188)), ((248, 194), (267, 213)), ((249, 89), (268, 108)), ((249, 90), (268, 109)), ((249, 170), (268, 189)), ((249, 195), (268, 214)), ((249, 248), (268, 267)), ((250, 90), (269, 109)), ((250, 91), (269, 110)), ((250, 170), (269, 189)), ((250, 171), (269, 190)), ((250, 196), (269, 215)), ((250, 249), (269, 268)), ((251, 91), (270, 110)), ((251, 92), (270, 111)), ((251, 171), (270, 190)), ((251, 172), (270, 191)), ((251, 197), (270, 216)), ((251, 250), (270, 269)), ((252, 92), (271, 111)), ((252, 93), (271, 112)), ((252, 172), (271, 191)), ((252, 173), (271, 192)), ((252, 198), (271, 217)), ((252, 251), (271, 270)), ((253, 93), (272, 112)), ((253, 94), (272, 113)), ((253, 173), (272, 192)), ((253, 174), (272, 193)), ((253, 199), (272, 218)), ((253, 252), (272, 271)), ((254, 94), (273, 113)), ((254, 95), (273, 114)), ((254, 174), (273, 193)), ((254, 175), (273, 194)), ((254, 200), (273, 219)), ((254, 253), (273, 272)), ((255, 96), (274, 115)), ((255, 175), (274, 194)), ((255, 176), (274, 195)), ((255, 201), (274, 220)), ((256, 97), (275, 116)), ((256, 176), (275, 195)), ((256, 177), (275, 196)), ((256, 202), (275, 221)), ((257, 98), (276, 117)), ((257, 177), (276, 196)), ((257, 178), (276, 197)), ((257, 203), (276, 222)), ((258, 99), (277, 118)), ((258, 178), (277, 197)), ((258, 179), (277, 198)), ((258, 204), (277, 223)), ((259, 100), (278, 119)), ((259, 179), (278, 198)), ((259, 180), (278, 199)), ((259, 205), (278, 224)), ((260, 101), (279, 120)), ((260, 180), (279, 199)), ((260, 181), (279, 200)), ((260, 206), (279, 225)), ((260, 235), (279, 254)), ((261, 51), (280, 70)), ((261, 76), (280, 95)), ((261, 77), (280, 96)), ((261, 102), (280, 121)), ((261, 103), (280, 122)), ((261, 181), (280, 200)), ((261, 182), (280, 201)), ((261, 207), (280, 226)), ((261, 235), (280, 254)), ((261, 236), (280, 255)), ((262, 52), (281, 71)), ((262, 77), (281, 96)), ((262, 78), (281, 97)), ((262, 103), (281, 122)), ((262, 104), (281, 123)), ((262, 182), (281, 201)), ((262, 183), (281, 202)), ((262, 208), (281, 227)), ((262, 236), (281, 255)), ((262, 237), (281, 256)), ((263, 53), (282, 72)), ((263, 78), (282, 97)), ((263, 79), (282, 98)), ((263, 104), (282, 123)), ((263, 105), (282, 124)), ((263, 183), (282, 202)), ((263, 184), (282, 203)), ((263, 209), (282, 228)), ((263, 237), (282, 256)), ((263, 238), (282, 257)), ((264, 54), (283, 73)), ((264, 79), (283, 98)), ((264, 80), (283, 99)), ((264, 105), (283, 124)), ((264, 106), (283, 125)), ((264, 184), (283, 203)), ((264, 185), (283, 204)), ((264, 210), (283, 229)), ((264, 238), (283, 257)), ((264, 239), (283, 258)), ((265, 80), (284, 99)), ((265, 81), (284, 100)), ((265, 106), (284, 125)), ((265, 107), (284, 126)), ((265, 185), (284, 204)), ((265, 186), (284, 205)), ((265, 211), (284, 230)), ((265, 239), (284, 258)), ((265, 240), (284, 259)), ((266, 81), (285, 100)), ((266, 107), (285, 126)), ((266, 108), (285, 127)), ((266, 186), (285, 205)), ((266, 212), (285, 231)), ((266, 240), (285, 259)), ((266, 241), (285, 260)), ((267, 108), (286, 127)), ((267, 213), (286, 232)), ((267, 241), (286, 260)), 
((309, 233), (328, 252)), ((310, 75), (329, 94)), ((310, 234), (329, 253)), ((311, 75), (330, 94)), ((311, 76), (330, 95)), ((311, 101), (330, 120)), ((311, 102), (330, 121)), ((311, 181), (330, 200)), ((311, 206), (330, 225)), ((311, 235), (330, 254)), ((311, 310), (330, 329)), ((312, 76), (331, 95)), ((312, 77), (331, 96)), ((312, 102), (331, 121)), ((312, 103), (331, 122)), ((312, 181), (331, 200)), ((312, 182), (331, 201)), ((312, 207), (331, 226)), ((312, 236), (331, 255)), ((312, 261), (331, 280)), ((312, 311), (331, 330)), ((313, 77), (332, 96)), ((313, 78), (332, 97)), ((313, 103), (332, 122)), ((313, 104), (332, 123)), ((313, 182), (332, 201)), ((313, 183), (332, 202)), ((313, 208), (332, 227)), ((313, 237), (332, 256)), ((313, 262), (332, 281)), ((313, 312), (332, 331)), ((314, 78), (333, 97)), ((314, 79), (333, 98)), ((314, 104), (333, 123)), ((314, 105), (333, 124)), ((314, 183), (333, 202)), ((314, 184), (333, 203)), ((314, 209), (333, 228)), ((314, 238), (333, 257)), ((314, 263), (333, 282)), ((314, 313), (333, 332)), ((315, 79), (334, 98)), ((315, 80), (334, 99)), ((315, 105), (334, 124)), ((315, 106), (334, 125)), ((315, 184), (334, 203)), ((315, 185), (334, 204)), ((315, 210), (334, 229)), ((315, 239), (334, 258)), ((315, 264), (334, 283)), ((315, 314), (334, 333)), ((316, 80), (335, 99)), ((316, 81), (335, 100)), ((316, 106), (335, 125)), ((316, 107), (335, 126)), ((316, 185), (335, 204)), ((316, 186), (335, 205)), ((316, 211), (335, 230)), ((316, 240), (335, 259)), ((316, 265), (335, 284)), ((316, 315), (335, 334)), ((317, 81), (336, 100)), ((317, 107), (336, 126)), ((317, 108), (336, 127)), ((317, 186), (336, 205)), ((317, 212), (336, 231)), ((317, 241), (336, 260)), ((317, 266), (336, 285)), ((317, 316), (336, 335)), ((318, 82), (337, 101)), ((318, 187), (337, 206)), ((318, 242), (337, 261)), ((318, 267), (337, 286)), ((319, 83), (338, 102)), ((319, 188), (338, 207)), ((319, 242), (338, 261)), ((319, 243), (338, 262)), ((319, 268), (338, 287)), ((320, 84), (339, 103)), ((320, 189), (339, 208)), ((320, 243), (339, 262)), ((320, 244), (339, 263)), ((320, 269), (339, 288)), ((321, 85), (340, 104)), ((321, 190), (340, 209)), ((321, 244), (340, 263)), ((321, 245), (340, 264)), ((322, 86), (341, 105)), ((322, 166), (341, 185)), ((322, 191), (341, 210)), ((322, 245), (341, 264)), ((322, 246), (341, 265)), ((323, 87), (342, 106)), ((323, 167), (342, 186)), ((323, 192), (342, 211)), ((323, 246), (342, 265)), ((323, 247), (342, 266)), ((324, 88), (343, 107)), ((324, 168), (343, 187)), ((324, 193), (343, 212)), ((324, 247), (343, 266)), ((324, 248), (343, 267)), ((325, 89), (344, 108)), ((325, 169), (344, 188)), ((325, 194), (344, 213)), ((325, 248), (344, 267)), ((325, 249), (344, 268)), ((326, 90), (345, 109)), ((326, 170), (345, 189)), ((326, 195), (345, 214)), ((326, 249), (345, 268)), ((326, 250), (345, 269)), ((327, 91), (346, 110)), ((327, 171), (346, 190)), ((327, 196), (346, 215)), ((327, 250), (346, 269)), ((327, 251), (346, 270)), ((328, 92), (347, 111)), ((328, 172), (347, 191)), ((328, 197), (347, 216)), ((328, 251), (347, 270)), ((328, 252), (347, 271)), ((329, 93), (348, 112)), ((329, 173), (348, 192)), ((329, 198), (348, 217)), ((329, 252), (348, 271)), ((329, 253), (348, 272)), ((330, 94), (349, 113)), ((330, 174), (349, 193)), ((330, 199), (349, 218)), ((330, 253), (349, 272)), ((330, 254), (349, 273)), ((331, 175), (350, 194)), ((331, 200), (350, 219)), ((331, 254), (350, 273)), ((331, 255), (350, 274)), ((332, 176), (351, 195)), ((332, 201), (351, 220)), ((332, 255), (351, 274)), ((332, 256), (351, 275)), ((333, 177), (352, 196)), ((333, 202), (352, 221)), ((333, 256), (352, 275)), ((333, 257), (352, 276)), ((334, 178), (353, 197)), ((334, 203), (353, 222)), ((334, 257), (353, 276)), ((334, 258), (353, 277)), ((335, 179), (354, 198)), ((335, 204), (354, 223)), ((335, 258), (354, 277)), ((335, 259), (354, 278)), ((336, 180), (355, 199)), ((336, 205), (355, 224)), ((336, 259), (355, 278)), ((336, 260), (355, 279)), ((337, 50), (356, 69)), ((337, 51), (356, 70)), ((337, 76), (356, 95)), ((337, 102), (356, 121)), ((337, 103), (356, 122)), ((337, 181), (356, 200)), ((337, 182), (356, 201)), ((337, 206), (356, 225)), ((337, 207), (356, 226)), ((337, 235), (356, 254)), ((337, 236), (356, 255)), ((337, 260), (356, 279)), ((337, 261), (356, 280)), ((337, 311), (356, 330)), ((337, 312), (356, 331)), ((338, 51), (357, 70)), ((338, 52), (357, 71)), ((338, 77), (357, 96)), ((338, 103), (357, 122)), ((338, 104), (357, 123)), ((338, 182), (357, 201)), ((338, 183), (357, 202)), ((338, 207), (357, 226)), ((338, 208), (357, 227)), ((338, 236), (357, 255)), ((338, 237), (357, 256)), ((338, 261), (357, 280)), ((338, 262), (357, 281)), ((338, 312), (357, 331)), ((338, 313), (357, 332)), ((339, 52), (358, 71)), ((339, 53), (358, 72)), ((339, 78), (358, 97)), ((339, 104), (358, 123)), ((339, 105), (358, 124)), ((339, 183), (358, 202)), ((339, 184), (358, 203)), ((339, 208), (358, 227)), ((339, 209), (358, 228)), ((339, 237), (358, 256)), ((339, 238), (358, 257)), ((339, 262), (358, 281)), ((339, 263), (358, 282)), ((339, 313), (358, 332)), ((339, 314), (358, 333)), ((340, 53), (359, 72)), ((340, 54), (359, 73)), ((340, 79), (359, 98)), ((340, 105), (359, 124)), ((340, 106), (359, 125)), ((340, 184), (359, 203)), ((340, 185), (359, 204)), ((340, 209), (359, 228)), ((340, 210), (359, 229)), ((340, 238), (359, 257)), ((340, 239), (359, 258)), ((340, 263), (359, 282)), ((340, 264), (359, 283)), ((340, 314), (359, 333)), ((340, 315), (359, 334)), ((341, 54), (360, 73)), ((341, 80), (360, 99)), ((341, 106), (360, 125)), ((341, 107), (360, 126)), ((341, 185), (360, 204)), ((341, 186), (360, 205)), ((341, 210), (360, 229)), ((341, 211), (360, 230)), ((341, 239), (360, 258)), ((341, 240), (360, 259)), ((341, 264), (360, 283)), ((341, 265), (360, 284)), ((341, 315), (360, 334)), ((341, 316), (360, 335)), ((342, 81), (361, 100)), ((342, 107), (361, 126)), ((342, 108), (361, 127)), ((342, 186), (361, 205)), ((342, 187), (361, 206)), ((342, 211), (361, 230)), ((342, 212), (361, 231)), ((342, 240), (361, 259)), ((342, 241), (361, 260)), ((342, 265), (361, 284)), ((342, 266), (361, 285)), ((342, 316), (361, 335)), ((342, 317), (361, 336)), ((343, 82), (362, 101)), ((343, 187), (362, 206)), ((343, 188), (362, 207)), ((343, 212), (362, 231)), ((343, 241), (362, 260)), ((343, 242), (362, 261)), ((343, 266), (362, 285)), ((343, 267), (362, 286)), ((343, 317), (362, 336)), ((343, 318), (362, 337)), ((344, 83), (363, 102)), ((344, 188), (363, 207)), ((344, 189), (363, 208)), ((344, 242), (363, 261)), ((344, 243), (363, 262)), ((344, 318), (363, 337)), ((344, 319), (363, 338)), ((345, 84), (364, 103)), ((345, 189), (364, 208)), ((345, 190), (364, 209)), ((345, 243), (364, 262)), ((345, 244), (364, 263)), ((345, 319), (364, 338)), ((345, 320), (364, 339)), ((346, 85), (365, 104)), ((346, 190), (365, 209)), ((346, 191), (365, 210)), ((346, 244), (365, 263)), ((346, 245), (365, 264)), ((346, 320), (365, 339)), ((346, 321), (365, 340)), ((347, 86), (366, 105)), ((347, 166), (366, 185)), ((347, 191), (366, 210)), ((347, 192), (366, 211)), ((347, 245), (366, 264)), ((347, 246), (366, 265)), ((347, 321), (366, 340)), ((347, 322), (366, 341)), ((348, 87), (367, 106)), ((348, 167), (367, 186)), ((348, 192), (367, 211)), ((348, 193), (367, 212)), ((348, 246), (367, 265)), ((348, 247), (367, 266)), ((348, 322), (367, 341)), ((348, 323), (367, 342)), ((349, 88), (368, 107)), ((349, 168), (368, 187)), ((349, 193), (368, 212)), ((349, 194), (368, 213)), ((349, 247), (368, 266)), ((349, 248), (368, 267)), ((349, 323), (368, 342)), ((349, 324), (368, 343)), ((350, 89), (369, 108)), ((350, 169), (369, 188)), ((350, 194), (369, 213)), ((350, 195), (369, 214)), ((350, 248), (369, 267)), ((350, 249), (369, 268)), ((350, 324), (369, 343)), ((350, 325), (369, 344)), ((351, 90), (370, 109)), ((351, 170), (370, 189)), ((351, 195), (370, 214)), ((351, 196), (370, 215)), ((351, 249), (370, 268)), ((351, 250), (370, 269)), ((351, 325), (370, 344)), ((351, 326), (370, 345)), ((351, 350), (370, 369)), ((352, 91), (371, 110)), ((352, 171), (371, 190)), ((352, 196), (371, 215)), ((352, 197), (371, 216)), ((352, 250), (371, 269)), ((352, 251), (371, 270)), ((352, 326), (371, 345)), ((352, 327), (371, 346)), ((352, 351), (371, 370)), ((353, 92), (372, 111)), ((353, 172), (372, 191)), ((353, 197), (372, 216)), ((353, 198), (372, 217)), ((353, 251), (372, 270)), ((353, 252), (372, 271)), ((353, 327), (372, 346)), ((353, 328), (372, 347)), ((353, 352), (372, 371)), ((354, 93), (373, 112)), ((354, 173), (373, 192)), ((354, 198), (373, 217)), ((354, 199), (373, 218)), ((354, 252), (373, 271)), ((354, 253), (373, 272)), ((354, 328), (373, 347)), ((354, 329), (373, 348)), ((354, 353), (373, 372)), ((355, 94), (374, 113)), ((355, 174), (374, 193)), ((355, 199), (374, 218)), ((355, 200), (374, 219)), ((355, 253), (374, 272)), ((355, 254), (374, 273)), ((355, 329), (374, 348)), ((355, 330), (374, 349)), ((355, 354), (374, 373)), ((356, 95), (375, 114)), ((356, 96), (375, 115)), ((356, 175), (375, 194)), ((356, 200), (375, 219)), ((356, 201), (375, 220)), ((356, 254), (375, 273)), ((356, 255), (375, 274)), ((356, 330), (375, 349)), ((356, 331), (375, 350)), ((357, 96), (376, 115)), ((357, 97), (376, 116)), ((357, 176), (376, 195)), ((357, 201), (376, 220)), ((357, 202), (376, 221)), ((357, 255), (376, 274)), ((357, 256), (376, 275)), ((357, 331), (376, 350)), ((357, 332), (376, 351)), ((358, 97), (377, 116)), ((358, 98), (377, 117)), ((358, 177), (377, 196)), ((358, 202), (377, 221)), ((358, 203), (377, 222)), ((358, 256), (377, 275)), ((358, 257), (377, 276)), ((358, 332), (377, 351)), ((358, 333), (377, 352)), ((359, 98), (378, 117)), ((359, 99), (378, 118)), ((359, 178), (378, 197)), ((359, 203), (378, 222)), ((359, 204), (378, 223)), ((359, 257), (378, 276)), ((359, 258), (378, 277)), ((359, 333), (378, 352)), ((359, 334), (378, 353)), ((360, 99), (379, 118)), ((360, 100), (379, 119)), ((360, 179), (379, 198)), ((360, 204), (379, 223)), ((360, 205), (379, 224)), ((360, 258), (379, 277)), ((360, 259), (379, 278)), ((360, 334), (379, 353)), ((360, 335), (379, 354)), ((361, 49), (380, 68)), ((361, 100), (380, 119)), ((361, 101), (380, 120)), ((361, 180), (380, 199)), ((361, 205), (380, 224)), ((361, 206), (380, 225)), ((361, 234), (380, 253)), ((361, 259), (380, 278)), ((361, 260), (380, 279)), ((361, 335), (380, 354)), ((361, 336), (380, 355)), ((362, 50), (381, 69)), ((362, 101), (381, 120)), ((362, 102), (381, 121)), ((362, 181), (381, 200)), ((362, 206), (381, 225)), ((362, 234), (381, 253)), ((362, 235), (381, 254)), ((362, 260), (381, 279)), ((362, 311), (381, 330)), ((362, 336), (381, 355)), ((362, 337), (381, 356)), ((390, 389), (409, 408)), ((391, 390), (410, 409)), ((392, 391), (411, 410)), ((393, 392), (412, 411)), ((394, 393), (413, 412))]
((24,       43     )
((49, 24), (68, 43))
 (234,49), (253,68)
 309, 233
 
 51
 76
 103
 182
 207
 236
 261
 312
 29  44  59 105  120 136 180 195
 
 [(28, 13), 
 (29, 13), 
 (44, 29), 
 (59, 29), 
 (59, 44), 
 (59, 29), 
 (96, 49), 
 (104, 44), 
 (104, 59), 
 (110, 96), 
 (112, 52), 
 (116, 55), 
 (119, 44), 
 (120, 29), 
 (120, 44), 
 (120, 104), 
 (122, 107), 
 (135, 28), 
 (135, 29), 
 (136, 44), 
 (136, 59), 
 (136, 104), 
 (136, 119), 
 (136, 120), 
 (142, 96), 
 (144, 51), 
 (145, 98), 
 (147, 55), 
 (147, 116), 
 (150, 136), 
 (151, 29), 
 (151, 44), 
 (151, 44), 
 (151, 59), 
 (151, 136), 
 (153, 107), 
 (179, 135), 
 (179, 43), 
 (180, 43), 
 (180, 58), 
 (180, 59), 
 (180, 104), 
 (180, 119), 
 (180, 104), 
 (180, 151), 
 (185, 140), 
 (186, 96), 
 (190, 146), 
 (192, 116), 
 (195, 29), 
 (195, 29), 
 (195, 44), 
 (195, 59), 
 (195, 59), 
 (195, 105), 
 (195, 120), 
 (195, 136), 
 (195, 136), 
 (195, 180), 
 (195, 180), 
 (196, 151), 
 (198, 107), (201, 96), (206, 55), (207, 55), (207, 117), (207, 147), (207, 148), (207, 191), (207, 192), (209, 28), (209, 135), (209, 119), (209, 135), (209, 180)]
 13: 2
 28 : 11
 44 : 12
 52 : 2
 55 : 2
 59 : 10
 96 : 4
 98 : 1
 49 : 1
 104 : 7
 107 : 1
 110 : 1
 112 : 1
 116 : 4
 119 : 8
 135 : 12
 140 : 1
 142 : 1
 144 : 2
 147 : 3
 150 : 8
 153 : 1
 179 : 8
 185 : 2
 190 : 1
 192 : 1
 195 : 12
 198 : 1
'''


'''
레드벨벳 러시안 룰렛(0.5)

[((178, 76), (197, 95)), ((179, 77), (198, 96)), ((180, 78), (199, 97)), ((181, 79), (200, 98)), ((182, 80), (201, 99)), ((183, 81), (202, 100)), ((184, 82), (203, 101)), ((185, 83), (204, 102)), ((186, 84), (205, 103)), ((187, 85), (206, 104)), ((188, 86), (207, 105)), ((189, 87), (208, 106)), ((190, 88), (209, 107)), ((191, 89), (210, 108)), ((192, 90), (211, 109)), ((193, 91), (212, 110)), ((193, 180), (212, 199)), ((194, 92), (213, 111)), ((194, 181), (213, 200)), ((195, 93), (214, 112)), ((196, 94), (215, 113)), ((197, 95), (216, 114)), ((198, 96), (217, 115)), ((199, 97), (218, 116)), ((200, 98), (219, 117)), ((201, 99), (220, 118)), ((202, 100), (221, 119)), ((203, 101), (222, 120)), ((204, 102), (223, 121)), ((205, 103), (224, 122)), ((206, 104), (225, 123)), ((207, 105), (226, 124)), ((208, 106), (227, 125)), 
((256, 177), (275, 196)), ((257, 178), (276, 197)), ((258, 77), (277, 96)), ((258, 179), (277, 198)), ((259, 78), (278, 97)), ((259, 180), (278, 199)), ((260, 78), (279, 97)), ((260, 79), (279, 98)), ((260, 180), (279, 199)), ((260, 181), (279, 200)), ((261, 80), (280, 99)), ((261, 181), (280, 200)), ((261, 182), (280, 201)), ((262, 81), (281, 100)), ((262, 182), (281, 201)), ((262, 183), (281, 202)), ((263, 82), (282, 101)), ((263, 183), (282, 202)), ((263, 184), (282, 203)), ((264, 83), (283, 102)), ((264, 184), (283, 203)), ((264, 185), (283, 204)), ((265, 84), (284, 103)), ((265, 185), (284, 204)), ((265, 186), (284, 205)), ((266, 85), (285, 104)), ((266, 186), (285, 205)), ((266, 187), (285, 206)), ((267, 86), (286, 105)), ((267, 187), (286, 206)), ((267, 188), (286, 207)), ((268, 87), (287, 106)), ((268, 188), (287, 207)), ((268, 189), (287, 208)), ((269, 88), (288, 107)), ((269, 189), (288, 208)), ((269, 190), (288, 209)), ((270, 89), (289, 108)), ((270, 190), (289, 209)), ((270, 191), (289, 210)), ((271, 90), (290, 109)), ((271, 191), (290, 210)), ((271, 192), (290, 211)), ((272, 91), (291, 110)), ((272, 192), (291, 211)), ((272, 193), (291, 212)), ((273, 92), (292, 111)), ((273, 193), (292, 212)), ((273, 194), (292, 213)), ((274, 93), (293, 112)), ((274, 194), (293, 213)), ((274, 195), (293, 214)), ((275, 94), (294, 113)), ((275, 195), (294, 214)), ((275, 196), (294, 215)), ((276, 95), (295, 114)), ((276, 196), (295, 215)), ((276, 197), (295, 216)), ((277, 96), (296, 115)), ((277, 198), (296, 217)), ((278, 97), (297, 116)), ((278, 199), (297, 218)), ((279, 98), (298, 117)), ((279, 200), (298, 219)), 
((310, 103), (329, 122)), ((310, 309), (329, 328)), ((311, 104), (330, 123)), ((311, 310), (330, 329)), ((312, 311), (331, 330)), ((313, 312), (332, 331)), 
((322, 309), (341, 328)), ((323, 310), (342, 329)), ((324, 311), (343, 330)), ((325, 312), (344, 331)), ((326, 313), (345, 332)), ((327, 314), (346, 333)), ((328, 315), (347, 334)), ((329, 316), (348, 335)), ((330, 317), (349, 336)), ((331, 318), (350, 337)), ((332, 319), (351, 338)), ((333, 320), (352, 339)), ((334, 321), (353, 340)), ((335, 310), (354, 329)), ((335, 322), (354, 341)), ((336, 78), (355, 97)), ((336, 180), (355, 199)), ((336, 259), (355, 278)), ((336, 311), (355, 330)), ((336, 323), (355, 342)), ((337, 79), (356, 98)), ((337, 181), (356, 200)), ((337, 260), (356, 279)), ((337, 312), (356, 331)), ((337, 324), (356, 343)), ((338, 313), (357, 332)), ((338, 325), (357, 344)), ((339, 314), (358, 333)), ((339, 326), (358, 345))]

44 103 148   60 179 186

44 59 103 111 149 179 186 194
55 114 159
-[(103, 44), 
- (111, 104), 
-(114, 55), 
-(148, 102), 
-(149, 44), 
-(150, 45), 
-(150, 104), 
-(159, 114), 
-(161, 56), 
-(179, 59), 
-(186, 178), 
-(194, 179), 
-(194, 45), 
-(194, 104), 
-(194, 150)]
'''



# print( round(( 211/365) * 76))
# print(round(( 211/365) * 178))
# print(round(( 211/365) * 256))
# print(round(( 211/365) * 103))
# print(round(( 211/365) * 310))
# print(round(( 211/365) * 322))


import numpy as np
c=[]
a = [[1+i,2+i] for i in range(10)]
[c.append([1+i, 5+i]) for i in range(10)]
print('c',c)
b = [[11+i,12+i] for i in range(10)]
print(a.append(b))
print('a+b',a + b)
print(a)
print(b)
print([1,2] in a)