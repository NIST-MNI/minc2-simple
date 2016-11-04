m2=require 'minc2_simple'

qqq=m2.minc2_file.new('/home/vfonov/data/viola03/models/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc')
-- qqq:open('/home/vfonov/mni/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc')

print(string.format("Loaded minc %dD file",qqq:ndim()))
dims=qqq:store_dims()

for i=0,(qqq:ndim()-1) do -- contrary to common LUA convention, it is 0-based
    print(string.format('Dimension %d length:%d id:%s start:%f step:%f',
        i,dims[i].length,dims[i].id,dims[i].start,dims[i].step))
end

ooo=m2.minc2_file.new()
-- will create file with same dimensions
ooo:define(dims, m2.MINC2_BYTE, m2.MINC2_FLOAT)
ooo:create('test_out.mnc')
ooo:copy_metadata(qqq)

buf=qqq:load_complete_volume(m2.MINC2_FLOAT)
ooo:save_complete_volume(buf,m2.MINC2_FLOAT)

ooo:close()



