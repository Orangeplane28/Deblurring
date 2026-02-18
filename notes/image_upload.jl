using Images
using ImageFeatures, TestImages, Images, ImageDraw, CoordinateTransformations, Rotations
using ImageFiltering, ImageView

img = testimage("mandrill");
println(typeof(img), "  ", size(img), "  ", eltype(img))
imgg = imfilter(img, Kernel.gaussian(3));
imgl = imfilter(img, Kernel.Laplacian());
imshow(img)

img = testimage("lighthouse")
img1 = Gray.(img)
rot = recenter(RotMatrix(5pi/6), [size(img1)...] .÷ 2)  # a rotation around the center
tform = rot ∘ Translation(-50, -40)
img2 = warp(img1, tform, axes(img1))
img_num = rand(4, 4)
imshow(img2)

img_gray_copy = Gray.(img_num) # construction
img_num_copy = Float64.(img_gray_copy) # construction

img_gray_view = colorview(Gray, img_num) # view
img_num_view = channelview(img_gray_view) # view