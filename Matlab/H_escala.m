% ============================================================
% Homografía CANÓNICA de ESCALA 2D 
% (u,v,w)^T = H * (x,y,1)^T     y     (x',y') = (u/w, v/w)
% ============================================================

clear; clc;

% --- 1) Factores de escala (elige los que quieras)
sx = 1.50;    % escala en x (sx>1 agranda, 0<sx<1 reduce)
sy = 0.75;    % escala en y

% --- 2) Matriz H de ESCALA alrededor del ORIGEN (0,0)
%     H_scale = [ sx  0  0
%                 0  sy  0
%                 0   0  1 ]
H = [ sx  0   0;
      0  sy   0;
      0   0   1];

% --- 3) Puntos de entrada (cartesianos). Cada columna es un punto (x;y).
X = [100   150    0;    % x
      80   120   50];   % y
N = size(X,2);

% --- 4) A homogéneas: (x,y,1)^T
Xh = [X; ones(1,N)];

% --- 5) Aplicar ESCALA sobre el ORIGEN
UVW = H * Xh;                 % (u,v,w)^T
Xp  = [UVW(1,:)./UVW(3,:);    % x' = u/w
       UVW(2,:)./UVW(3,:)];   % y' = v/w

% --- 6) Resultados (escala sobre el ORIGEN)
disp('Matriz H (escala sobre ORIGEN):');
disp(H);

disp('Puntos de entrada X (cartesianos):');
disp(X);

disp('Puntos SALIDA Xp = (x'', y'') tras escalar sobre ORIGEN:');
disp(Xp);

for i = 1:N
    fprintf('ORIGEN -> Punto %d: (x,y)=(%g,%g)  ->  (x'',y'')=(%g,%g)\n', ...
        i, X(1,i), X(2,i), Xp(1,i), Xp(2,i));
end

% ============================================================
% ESCALA alrededor de un CENTRO arbitrario C = (cx, cy)
% H_centro = T(+C) * S(sx,sy) * T(-C)
% ============================================================

% --- 7) Centro de escala arbitrario
cx = 120;    % coord x del centro
cy =  90;    % coord y del centro

% --- 8) Traslaciones (homogéneas)
T_minusC = [1 0 -cx;
            0 1 -cy;
            0 0   1];   % lleva C al origen

T_plusC  = [1 0  cx;
            0 1  cy;
            0 0   1];   % regresa del origen a C

% --- 9) ESCALA alrededor de C: Hc = T(+C) * S * T(-C)
S = H;                 % reutilizamos la S(sx,sy) definida como "H" arriba
Hc = T_plusC * S * T_minusC;

% --- 10) Aplicar ESCALA alrededor de C
UVWc = Hc * Xh;
XpC  = [UVWc(1,:)./UVWc(3,:);    % x' alrededor de C
        UVWc(2,:)./UVWc(3,:)];   % y' alrededor de C

% --- 11) Resultados (escala sobre el centro C)
disp(' ');
disp('Centro de escala C = (cx, cy):');
disp([cx, cy]);

disp('Matriz Hc (escala sobre el CENTRO C):');
disp(Hc);

disp('Puntos SALIDA XpC = (x'', y'') tras escalar sobre CENTRO C:');
disp(XpC);

for i = 1:N
    fprintf('CENTRO C -> Punto %d: (x,y)=(%g,%g)  ->  (x'',y'')=(%g,%g)\n', ...
        i, X(1,i), X(2,i), XpC(1,i), XpC(2,i));
end


