%% Slide 25 — Radial lens distortion

clear; close all; clc;

k2 = 0.02;
k1_barrel = -0.28;
k1_pincushion = 0.22;
grid_values = linspace(-0.8,0.8,11);
line_samples = linspace(-0.8,0.8,160);
test_point = [0.65; 0.45];

[barrel_point,barrel_scale] = distort_points(test_point,k1_barrel,k2);
[pincushion_point,pincushion_scale] = distort_points(test_point,k1_pincushion,k2);

fprintf('RADIAL LENS DISTORTION\n');
fprintf('Ideal normalized point:       (%.4f, %.4f)\n',test_point(1),test_point(2));
fprintf('Barrel-distorted point:       (%.4f, %.4f), radial scale = %.4f\n', ...
    barrel_point(1),barrel_point(2),barrel_scale);
fprintf('Pincushion-distorted point:   (%.4f, %.4f), radial scale = %.4f\n', ...
    pincushion_point(1),pincushion_point(2),pincushion_scale);

figure('Color','w','Name','Slide 25: radial lens distortion');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

plot_grid(0,0,grid_values,line_samples,1,'Ideal normalized grid');
plot_grid(k1_barrel,k2,grid_values,line_samples,2, ...
    sprintf('Barrel distortion: k_1 = %.2f',k1_barrel));
plot_grid(k1_pincushion,k2,grid_values,line_samples,3, ...
    sprintf('Pincushion distortion: k_1 = %.2f',k1_pincushion));

radial_distance = linspace(0,1.15,300);
barrel_radial_scale = 1 + k1_barrel*radial_distance.^2 + k2*radial_distance.^4;
pincushion_radial_scale = 1 + k1_pincushion*radial_distance.^2 + k2*radial_distance.^4;

nexttile(4);
plot(radial_distance,ones(size(radial_distance)),'k--','LineWidth',1.4); hold on;
plot(radial_distance,barrel_radial_scale,'LineWidth',2);
plot(radial_distance,pincushion_radial_scale,'LineWidth',2);
grid on;
xlabel('Normalized radius r'); ylabel('Radial scale');
title('Scale factor versus distance from the center');
legend('Ideal','Barrel','Pincushion','Location','best');

function plot_grid(k1,k2,grid_values,line_samples,panel,panel_title)
    nexttile(panel);
    hold on; grid on; axis equal;
    for value = grid_values
        horizontal = [line_samples; value*ones(size(line_samples))];
        vertical = [value*ones(size(line_samples)); line_samples];
        horizontal_d = distort_points(horizontal,k1,k2);
        vertical_d = distort_points(vertical,k1,k2);
        plot(horizontal_d(1,:),horizontal_d(2,:),'Color',[0.00 0.45 0.70]);
        plot(vertical_d(1,:),vertical_d(2,:),'Color',[0.95 0.45 0.10]);
    end
    xlim([-1.05 1.05]); ylim([-1.05 1.05]);
    set(gca,'YDir','reverse');
    xlabel('x'); ylabel('y'); title(panel_title);
end

function [distorted,radial_scale] = distort_points(points,k1,k2)
    r2 = sum(points.^2,1);
    radial_scale = 1 + k1*r2 + k2*r2.^2;
    distorted = points .* radial_scale;
end
