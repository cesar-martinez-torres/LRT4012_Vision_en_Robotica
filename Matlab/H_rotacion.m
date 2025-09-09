% ============================================================
% Homografía CANÓNICA de ROTACIÓN 2D 
% (u,v,w)^T = H * (x,y,1)^T     y     (x',y') = (u/w, v/w)
% ============================================================

clear; clc;

% --- 1) Ángulo de rotación (elige grados); positivo = antihorario
theta_deg = 30;                  % grados
theta = (pi/180) * theta_deg;    % a radianes 

% --- 2) Matriz H de rotación alrededor del ORIGEN (0,0)
%     H_rot = [ cosθ  -sinθ   0
%               sinθ   cosθ   0
%                 0      0    1 ]
H = [ cos(theta)  -sin(theta)   0;
      sin(theta)   cos(theta)   0;
           0             0      1];

% --- 3) Puntos de entrada (cartesianos). Cada columna es un punto (x;y).
%        Puedes modificar, agregar o quitar puntos.
X = [100   150    0;    % x
      80   120   50];   % y
N = size(X,2);

% --- 4) Convertir a homogéneas: (x,y,1)^T
Xh = [X; ones(1,N)];

% --- 5) Aplicar la homografía de ROTACIÓN alrededor del ORIGEN
UVW = H * Xh;                 % (u,v,w)^T
Xp  = [UVW(1,:)./UVW(3,:);    % x' = u/w
       UVW(2,:)./UVW(3,:)];   % y' = v/w

% --- 6) Mostrar resultados (rotación alrededor del origen)
disp('Matriz H (rotación sobre el ORIGEN):');
disp(H);

disp('Puntos de entrada X (cartesianos):');
disp(X);

disp('Puntos SALIDA Xp = (x'', y'') tras rotar sobre el ORIGEN:');
disp(Xp);

for i = 1:N
    fprintf('ORIGEN -> Punto %d: (x,y)=(%g,%g)  ->  (x'',y'')=(%g,%g)\n', ...
        i, X(1,i), X(2,i), Xp(1,i), Xp(2,i));
end

% ============================================================
% ROTACIÓN alrededor de un CENTRO arbitrario C = (cx, cy)
% H_centro = T(+C) * R(θ) * T(-C)
% ============================================================

% --- 7) Define centro de rotación arbitrario
cx = 120;    % coordenada x del centro
cy =  90;    % coordenada y del centro

% --- 8) Matrices de traslación (a homogéneas)
% T_minusC: traslada el centro C al origen
T_minusC = [1 0 -cx;
            0 1 -cy;
            0 0   1];

% T_plusC: regresa del origen al centro C
T_plusC  = [1 0 cx;
            0 1 cy;
            0 0  1];

% --- 9) Rotación alrededor de C: Hc = T(+C) * R * T(-C)
R = H;                  % reutilizamos la R(θ) definida como "H" arriba
Hc = T_plusC * R * T_minusC;

% --- 10) Aplicar rotación alrededor de C
UVWc = Hc * Xh;
XpC  = [UVWc(1,:)./UVWc(3,:);    % x' alrededor de C
        UVWc(2,:)./UVWc(3,:)];   % y' alrededor de C

% --- 11) Mostrar resultados (rotación alrededor del centro C)
disp(' ');
disp('Centro de rotación C = (cx, cy):');
disp([cx, cy]);

disp('Matriz Hc (rotación sobre el CENTRO C):');
disp(Hc);

disp('Puntos SALIDA XpC = (x'', y'') tras rotar sobre el CENTRO C:');
disp(XpC);

for i = 1:N
    fprintf('CENTRO C -> Punto %d: (x,y)=(%g,%g)  ->  (x'',y'')=(%g,%g)\n', ...
        i, X(1,i), X(2,i), XpC(1,i), XpC(2,i));
end


