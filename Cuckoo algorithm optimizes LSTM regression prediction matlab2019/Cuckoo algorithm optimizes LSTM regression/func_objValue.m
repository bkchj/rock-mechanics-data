function [ result ] = func_objValue( pop )
%FUNC_OBJVALUE ����Ŀ�꺯��
objValue =  sum(pop.^2,2);
result = objValue ;
end

