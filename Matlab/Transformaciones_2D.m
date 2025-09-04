%% Transformaciones 2D 
syms p2 p1 T11 T12 T21 T22 T x2 y2 x1 y1
T=[T11 T12;T21 T22];
p2=[x2;y2]
p1=[x1;y1]
ec1=p2==T*p1
%% Escalar Imagen
% x=ax1 y=by1 
syms a b
T11=a
T12=0
T21=0
T22=b
eval(T)
eval(ec1)
a=4
b=5
eval(T)
eval(ec1)
%% Traslación

