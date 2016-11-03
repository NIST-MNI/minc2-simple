m2=require 'minc2_simple'

qqq=m2.minc2_file.new()
qqq:open('/home/vfonov/mni/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc')

print("Loaded minc file")
print(qqq)
print(qqq:ndim())

for i=0,(qqq:ndim()-1) do
    print(string.format('Dimension %d length:%d id:%s start:%f step:%f',
        i,qqq:dims()[i].length,qqq:dims()[i].id,qqq:dims()[i].start,qqq:dims()[i].step))
end
