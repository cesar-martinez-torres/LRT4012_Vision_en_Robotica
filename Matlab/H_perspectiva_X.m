% ============================================================
% Homografía CANÓNICA de PERSPECTIVA simple en X
% (u,v,w)^T = H * (x,y,1)^T      y       (x',y') = (u/w, v/w)
% ============================================================

clear; clc;

% --- 1) Parámetro de perspectiva en X
%     p > 0  -> comprime conforme x aumenta (punto de fuga hacia +x)
%     p < 0  -> comprime conforme x disminuye (punto de fuga hacia -x)
p = 0.002;   % prueba con 0.001 ~ 0.005 (valores muy grandes distorsionan mucho)

% --- 2) Matriz H proyectiva (perspectiva en X) respecto al ORIGEN

%     Ecuaciones:
%        u = x
%        v = y
%        w = p*x + 1
%     Deshomogeneización:
%        x' = u/w = x / (p*x + 1),     y' = v/w = y / (p*x + 1)
H = [ 1   0   0;
      0   1   0;
      p   0   1];

% --- 3) Puntos de entrada (cada columna es (x;y))
X = [  -500   -200      0    200    500;   % x
         80    120     50     20   -30 ];  % y
N = size(X,2);

% --- 4) A homogéneas
Xh = [X; ones(1,N)];

% --- 5) Aplicar perspectiva en X (respecto al ORIGEN)
UVW = H * Xh;                      % (u,v,w)^T
Xp  = [ UVW(1,:)./UVW(3,:);        % x' = u/w = x/(p*x+1)
        UVW(2,:)./UVW(3,:) ];      % y' = v/w = y/(p*x+1)

% --- 6) Mostrar resultados
disp('Matriz H (perspectiva simple en X, respecto al ORIGEN):');
disp(H);
disp('Puntos de entrada X:'); disp(X);
disp('UVW (u;v;w) = H * (x;y;1):'); disp(UVW);
disp('Puntos salida Xp = (x'',y'') = (u/w, v/w):'); disp(Xp);

% - Es PROYECTIVA: h31 = p ≠ 0  =>  w = p*x + 1 depende de x.
% - "Línea singular" (va al infinito): p*x + 1 = 0  ->  x = -1/p
%   (cualquier punto con esa x mapea a w=0 -> ideal/proyectivo).
% - Si p>0 y x→(+∞), w→(+∞) y (x',y')→(1/p, 0) en escala relativa -> compresión hacia la derecha.
%   Si p<0, compresión hacia la izquierda.

% ============================================================
% PERSPECTIVA en X alrededor de un CENTRO arbitrario C = (cx, cy)
% H_c = T(+C) * H_px * T(-C)
% (conjugación para cambiar el "eje" respecto al que se mide x)
% ============================================================

% --- 8) Centro C
cx = 120;   % mover el "eje" vertical de referencia a x = cx
cy =  90;   % (cy no afecta w en este modelo, pero sí la traslación final)

% Traslaciones homogéneas
T_minusC = [1 0 -cx;
            0 1 -cy;
            0 0   1];
T_plusC  = [1 0  cx;
            0 1  cy;
            0 0   1];

% --- 9) Matriz H alrededor de C
Hc = T_plusC * H * T_minusC;

% --- 10) Aplicar Hc
UVWc = Hc * Xh;
XpC  = [ UVWc(1,:)./UVWc(3,:);    % x' alrededor de C
         UVWc(2,:)./UVWc(3,:) ];

disp(' ');
disp('Centro C = (cx, cy):'); disp([cx, cy]);
disp('Matriz Hc (perspectiva en X alrededor de C):');
disp(Hc);
disp('Salida XpC = (x'',y'') con centro C:'); disp(XpC);

% --- 11) Fórmula analítica de w con centro C:
% Conjugación con T(-C) hace que w = p*(x - cx) + 1
%   => línea singular:  p*(x - cx) + 1 = 0  ->  x = cx - 1/p
%    (vertical por x = cx - 1/p)
%
% Para verificar rápidamente:
w_origin  = p * X(1,:) + 1;          % respecto al origen
w_centerC = p * (X(1,:) - cx) + 1;   % respecto al centro C (solo para referencia)
disp('w (origen):   p*x + 1 ='); disp(w_origin);
disp('w (centro C): p*(x-cx) + 1 ='); disp(w_centerC);


