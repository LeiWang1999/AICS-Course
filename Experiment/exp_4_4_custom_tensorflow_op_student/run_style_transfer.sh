rm *.pyc
rm *.jpg

python stu_upload/transform_cpu.py ./images/chicago.jpg ./models/pb_models/udnie.pb  ./stu_upload/udnie_power_diff.pb ./stu_upload/udnie_power_diff_numpy.pb
