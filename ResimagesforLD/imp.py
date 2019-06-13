import scipy.misc
from scipy import misc
from scipy.misc.pilutil import Image
 
im = Image.open('car3_lp.jpg')
im_array = scipy.misc.fromimage(im)
im_inverse = 255 - im_array
im_result = scipy.misc.toimage(im_inverse)
misc.imsave('zzcar3_lp.jpg',im_result)