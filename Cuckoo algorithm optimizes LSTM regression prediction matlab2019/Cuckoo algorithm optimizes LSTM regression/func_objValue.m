function [ result ] = func_objValue( pop )
%FUNC_OBJVALUE 计算目标函数
objValue =  sum(pop.^2,2);
result = objValue ;
end

