#! /usr/bin/env th

require 'minc2_simple'
-- require 'torch'
require 'math'

local xfm_file=arg[1]
local ref=arg[2]



--minc2_xfm
--print(arg)

local xfm=minc2_xfm.new(xfm_file)

local edges={ {-96,-132, -78},
              {97, 97, 115}
            }

if ref then
  --print("Using:"..ref)
  
else
  --print("Using xfm directly")
end


local x,y,z
local max_dist=0.0
for x=1,2 do
    for y=1,2 do
        for z=1,2 do
            local p_in={ edges[x][1],edges[y][2],edges[z][3] }
            local p_out=xfm:transform_point(p_in)
            
            local dist=math.sqrt( (p_in[1]-p_out[1])^2+
                                  (p_in[2]-p_out[2])^2+
                                  (p_in[3]-p_out[3])^2)
            if dist>max_dist then
                max_dist=dist
            end
            
        end
    end
end

print("Max dist:"..max_dist)


