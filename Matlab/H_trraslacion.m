% ============================================================
% Homografía de TRASLACIÓN 2D 
% (u,v,w)^T = H * (x,y,1)^T     y     (x',y') = (u/w, v/w)
% ============================================================

clear; clc;

% --- 1) Parámetros de traslación (elige los que quieras)
tx = 40;     % desplazamiento en x
ty = -15;    % desplazamiento en y

% --- 2) Matriz H canónica de traslación (afín: sin perspectiva)

H = [1 0 tx;
     0 1 ty;
     0 0  1];

% --- 3) Puntos de entrada (cartesianos). Cada columna es un punto (x;y).
%        Puedes cambiar, agregar o quitar puntos.
X = [100  150   0;    % x
      80  120  50];   % y
N = size(X,2);

% --- 4) Convertir a homogéneas (añadir fila de unos): (x,y,1)^T
Xh = [X; ones(1,N)];

% --- 5) Aplicar la homografía: (u,v,w)^T = H * (x,y,1)^T
UVW = H * Xh;    % UVW(1,:)=u, UVW(2,:)=v, UVW(3,:)=w

% --- 6) Deshomogeneizar: (x',y') = (u/w, v/w)
Xp = [ UVW(1,:)./UVW(3,:);    % x'
       UVW(2,:)./UVW(3,:) ];  % y'

% --- 7) Mostrar resultados
disp('Matriz H (traslación):');
disp(H);

disp('Puntos de entrada X (cartesianos):');
disp(X);

disp('Puntos en homogéneas de SALIDA (u; v; w):');
disp(UVW);

disp('Puntos de salida Xp = (x'', y'') (deshomogeneizados):');
disp(Xp);

% --- 8) Interpretación:
% En traslación pura, w = 1 para todos los puntos (fila 3 de UVW es todo unos).
% Por tanto:
%   u = x + tx,    v = y + ty,    w = 1
%   x' = u/w = x + tx
%   y' = v/w = y + ty

% --- 9) Comprobación directa elemento a elemento (solo para visualizar)
%      (no es necesario, pero ayuda a entender u,v,w y x',y')
for i = 1:N
    x  = X(1,i); y  = X(2,i);
    u  = UVW(1,i); v = UVW(2,i); w = UVW(3,i);
    xp = Xp(1,i);  yp = Xp(2,i);
    fprintf('Punto %d:  (x,y)=(%g,%g) -> (u,v,w)=(%g,%g,%g) -> (x'',y'')=(%g,%g)\n', ...
            i, x, y, u, v, w, xp, yp);
end

% --- 10) Nota:
% Si cambias tx, ty verás que simplemente se SUMAN a x e y.
% Al ser afín (h31=h32=0), no hay perspectiva ni puntos de fuga.
