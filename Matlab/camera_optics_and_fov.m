%% Slide 23 — Focal length and sensor size set field of view

clear; close all; clc;

sensor_width = 6.4;
sensor_height = 4.8;
focal_lengths = linspace(2.5,35,400);
selected_f = [4 8 16];
working_distance = 2.0;

fov_x = 2 * atan(sensor_width ./ (2*focal_lengths));
fov_y = 2 * atan(sensor_height ./ (2*focal_lengths));
selected_fov_x = 2 * atan(sensor_width ./ (2*selected_f));
selected_fov_y = 2 * atan(sensor_height ./ (2*selected_f));
coverage_x = 2 * working_distance .* tan(selected_fov_x/2);
coverage_y = 2 * working_distance .* tan(selected_fov_y/2);

fprintf('FIELD OF VIEW AND COVERAGE\n');
fprintf('Sensor dimensions: %.1f mm x %.1f mm\n',sensor_width,sensor_height);
fprintf('f [mm]   FOV_x [deg]   FOV_y [deg]   C_x at %.1f m [m]   C_y at %.1f m [m]\n', ...
    working_distance,working_distance);
for i = 1:numel(selected_f)
    fprintf('%6.1f     %9.2f      %9.2f          %9.3f          %9.3f\n', ...
        selected_f(i),rad2deg(selected_fov_x(i)),rad2deg(selected_fov_y(i)), ...
        coverage_x(i),coverage_y(i));
end

figure('Color','w','Name','Slides 23-24: optics and field of view');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(focal_lengths,rad2deg(fov_x),'LineWidth',2); hold on;
plot(focal_lengths,rad2deg(fov_y),'LineWidth',2);
plot(selected_f,rad2deg(selected_fov_x),'ko','MarkerFaceColor',[0.95 0.45 0.10]);
grid on;
xlabel('Focal length [mm]'); ylabel('Angular field of view [deg]');
title('Field of view versus focal length');
legend('Horizontal FOV','Vertical FOV','Selected lenses','Location','northeast');

nexttile;
hold on; grid on; axis equal;
coverage_colors = lines(numel(selected_f));
for i = 1:numel(selected_f)
    half_width = coverage_x(i)/2;
    plot([0 working_distance],[0 half_width],'-','Color',coverage_colors(i,:), ...
        'LineWidth',1.8,'DisplayName',sprintf('f = %.0f mm',selected_f(i)));
    plot([0 working_distance],[0 -half_width],'-','Color',coverage_colors(i,:), ...
        'LineWidth',1.8,'HandleVisibility','off');
end
xlabel('Optical-axis distance [m]'); ylabel('Horizontal position [m]');
title(sprintf('Horizontal coverage at Z = %.1f m',working_distance));
legend('Location','best');

%% Slide 24 — Thin-lens equation

focus_f = 8.0;
object_distance = linspace(1.05*focus_f,2000,600);
image_distance = 1 ./ (1/focus_f - 1./object_distance);
magnification = -image_distance ./ object_distance;

selected_zo = [50 100 500 2000];
selected_zi = 1 ./ (1/focus_f - 1./selected_zo);
selected_m = -selected_zi ./ selected_zo;

fprintf('\nTHIN-LENS CALCULATION\n');
fprintf('f = %.1f mm\n',focus_f);
fprintf('z_o [mm]   z_i [mm]   magnification\n');
for i = 1:numel(selected_zo)
    fprintf('%8.1f   %8.4f      %9.5f\n',selected_zo(i),selected_zi(i),selected_m(i));
end

nexttile;
plot(object_distance,image_distance,'LineWidth',2); hold on;
yline(focus_f,'k--','f');
plot(selected_zo,selected_zi,'o','MarkerFaceColor',[0.95 0.45 0.10]);
grid on;
xlabel('Object distance z_o [mm]'); ylabel('Image distance z_i [mm]');
title(sprintf('Focus for f = %.1f mm',focus_f));
ylim([0 80]);

nexttile;
semilogx(object_distance,abs(magnification),'LineWidth',2); hold on;
semilogx(selected_zo,abs(selected_m),'o','MarkerFaceColor',[0.00 0.55 0.80]);
grid on;
xlabel('Object distance z_o [mm]'); ylabel('|m| = |z_i/z_o|');
title('Magnification decreases with distance');
