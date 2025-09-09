% ============================================================
% Homografías CANÓNICAS de CIZALLA 2D
% (u,v,w)^T = H * (x,y,1)^T     y     (x',y') = (u/w, v/w)
% ============================================================

clear; clc;

% --- 1) Parámetros de cizalla
kx = 0.40;   % cizalla EN X (u depende de y): u = x + kx*y
ky = -0.25;  % cizalla EN Y (v depende de x): v = y + ky*x

% --- 2) Matrices H canónicas (afines: sin perspectiva, h31=h32=0)
% Cizalla en X : desplaza X proporcionalmente a Y

Hx = [1  kx  0;
      0   1  0;
      0   0  1];

% Cizalla en Y : desplaza Y proporcionalmente a X

Hy = [1   0  0;
      ky  1  0;
       0  0  1];

% --- 3) Puntos de prueba (cartesianos). Cada columna es un punto (x;y).
X = [100   150     0   60;   % x
      80   120    50  -20];  % y
N = size(X,2);

% --- 4) A homogéneas
Xh = [X; ones(1,N)];

% --- 5) Aplicar CIZALLA EN X
UVWx = Hx * Xh;                    % (u,v,w) 
Xp_x = [UVWx(1,:)./UVWx(3,:);      % x' = u/w  (aquí w=1)
        UVWx(2,:)./UVWx(3,:)];     % y' = v/w

% --- 6) Aplicar CIZALLA EN Y
UVWy = Hy * Xh;                    % (u,v,w) 
Xp_y = [UVWy(1,:)./UVWy(3,:);      % x' = u/w
        UVWy(2,:)./UVWy(3,:)];     % y' = v/w

% --- 7) Mostrar resultados
disp('Matriz Hx (cizalla en X):'); disp(Hx);
disp('Matriz Hy (cizalla en Y):'); disp(Hy);

disp('Puntos de entrada X (cartesianos):'); disp(X);

disp('Salida con cizalla en X  -> Xp_x = (x'', y''):'); disp(Xp_x);
disp('Salida con cizalla en Y  -> Xp_y = (x'', y''):'); disp(Xp_y);

for i = 1:N
    fprintf('Cizalla-X -> P%d: (x,y)=(%7.2f,%7.2f)  ->  (x'',y'')=(%7.2f,%7.2f)\n', ...
        i, X(1,i), X(2,i), Xp_x(1,i), Xp_x(2,i));
end
for i = 1:N
    fprintf('Cizalla-Y -> P%d: (x,y)=(%7.2f,%7.2f)  ->  (x'',y'')=(%7.2f,%7.2f)\n', ...
        i, X(1,i), X(2,i), Xp_y(1,i), Xp_y(2,i));
end

% ============================================================
% 8) CIZALLA alrededor de un CENTRO arbitrario C = (cx, cy)
%     Hc = T(+C) * Hshear * T(-C)
% ============================================================

% --- Centro de referencia
cx = 120;
cy =  90;

% Traslaciones homogéneas
T_minusC = [1 0 -cx;
            0 1 -cy;
            0 0   1];
T_plusC  = [1 0  cx;
            0 1  cy;
            0 0   1];

% Cizalla en X alrededor de C
HxC = T_plusC * Hx * T_minusC;

% Cizalla en Y alrededor de C
HyC = T_plusC * Hy * T_minusC;

% Aplicar
UVWxC = HxC * Xh;  Xp_xC = [UVWxC(1,:)./UVWxC(3,:); UVWxC(2,:)./UVWxC(3,:)];
UVWyC = HyC * Xh;  Xp_yC = [UVWyC(1,:)./UVWyC(3,:); UVWyC(2,:)./UVWyC(3,:)];

disp(' ');
disp('Centro C de referencia para cizalla:'); disp([cx, cy]);
disp('HxC (cizalla en X alrededor de C):'); disp(HxC);
disp('HyC (cizalla en Y alrededor de C):'); disp(HyC);

disp('Salida con Cizalla-X alrededor de C -> Xp_xC:'); disp(Xp_xC);
disp('Salida con Cizalla-Y alrededor de C -> Xp_yC:'); disp(Xp_yC);

