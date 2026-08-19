%% Slide 18 — Perspective projection in the pinhole model

clear; close all; clc;

f_metric = 0.080;
depths = [0.50 1.00 2.00];
half_size = 0.08;
square_xy = half_size * [-1 -1; 1 -1; 1 1; -1 1; -1 -1];
depth_colors = lines(numel(depths));

metric_projection = cell(numel(depths),1);
for i = 1:numel(depths)
    Zc = depths(i);
    metric_projection{i} = f_metric * square_xy / Zc;
end

fprintf('PINHOLE PROJECTION\n');
fprintf('x = f X_c/Z_c, y = f Y_c/Z_c\n');
fprintf('f = %.3f m, object width = %.3f m\n', f_metric, 2*half_size);
for i = 1:numel(depths)
    projected_width_mm = 1000 * (max(metric_projection{i}(:,1)) - min(metric_projection{i}(:,1)));
    fprintf('Z_c = %.2f m -> projected width = %.3f mm\n', depths(i), projected_width_mm);
end

figure('Color','w','Name','Slides 18-22: camera projection pipeline');
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile;
hold on; grid on; axis equal;
for i = 1:numel(depths)
    plot3(square_xy(:,1), square_xy(:,2), depths(i)*ones(size(square_xy,1),1), ...
        '-o','Color',depth_colors(i,:),'LineWidth',1.6, ...
        'DisplayName',sprintf('Z_c = %.2f m',depths(i)));
end
plot3(0,0,0,'kp','MarkerFaceColor','k','MarkerSize',11,'DisplayName','Pinhole');
xlabel('X_c [m]'); ylabel('Y_c [m]'); zlabel('Z_c [m]');
title('Identical objects in the camera frame');
view(35,24); legend('Location','best','FontSize',8);

nexttile;
hold on; grid on; axis equal;
for i = 1:numel(depths)
    plot(1000*metric_projection{i}(:,1),1000*metric_projection{i}(:,2), ...
        '-o','Color',depth_colors(i,:),'LineWidth',1.6, ...
        'DisplayName',sprintf('Z_c = %.2f m',depths(i)));
end
set(gca,'YDir','reverse');
xlabel('x [mm]'); ylabel('y [mm]');
title('Metric image-plane projection');
legend('Location','best','FontSize',8);

%% Slide 19 — Extrinsics place the camera in the robot world

Pw = [0.0  0.0 3.0;
      0.8  0.0 3.2;
      0.0  0.7 3.5;
     -0.6 -0.3 4.0]';

Cw = [0.40; -0.20; 0.30];
yaw = deg2rad(8);
pitch = deg2rad(-5);
roll = deg2rad(3);

Rx = [1 0 0; 0 cos(roll) -sin(roll); 0 sin(roll) cos(roll)];
Ry = [cos(pitch) 0 sin(pitch); 0 1 0; -sin(pitch) 0 cos(pitch)];
Rz = [cos(yaw) -sin(yaw) 0; sin(yaw) cos(yaw) 0; 0 0 1];

R_wc = Rz * Ry * Rx;
R_cw = R_wc';
t_cw = -R_cw * Cw;
Pc = R_cw * Pw + t_cw;

fprintf('\nWORLD-TO-CAMERA TRANSFORMATION\n');
fprintf('R_cw^T R_cw:\n');
disp(R_cw' * R_cw);
fprintf('det(R_cw) = %.12f\n',det(R_cw));
fprintf('t_cw [m]:\n');
disp(t_cw);
fprintf('P_c columns [m]:\n');
disp(Pc);

nexttile;
hold on; grid on; axis equal;
plot3(Pw(1,:),Pw(2,:),Pw(3,:),'ko','MarkerFaceColor',[0.95 0.45 0.10], ...
    'MarkerSize',7,'DisplayName','World landmarks');
plot3(Cw(1),Cw(2),Cw(3),'kp','MarkerFaceColor','k','MarkerSize',11, ...
    'DisplayName','Camera center');
axis_length = 0.45;
axis_colors = {'r','g','b'};
axis_names = {'Camera X-axis','Camera Y-axis','Camera Z-axis'};
for j = 1:3
    quiver3(Cw(1),Cw(2),Cw(3),axis_length*R_wc(1,j),axis_length*R_wc(2,j), ...
        axis_length*R_wc(3,j),axis_colors{j},'LineWidth',1.8,'MaxHeadSize',0.5, ...
        'DisplayName',axis_names{j});
end
xlabel('X_w [m]'); ylabel('Y_w [m]'); zlabel('Z_w [m]');
title('Camera pose in the world');
view(35,22); legend('Location','southwest','FontSize',7);

nexttile;
hold on; grid on; axis equal;
plot3(Pc(1,:),Pc(2,:),Pc(3,:),'o','Color',[0.00 0.35 0.65], ...
    'MarkerFaceColor',[0.00 0.55 0.80],'MarkerSize',7);
plot3(0,0,0,'kp','MarkerFaceColor','k','MarkerSize',11);
quiver3(0,0,0,0.45,0,0,'r','LineWidth',1.8);
quiver3(0,0,0,0,0.45,0,'g','LineWidth',1.8);
quiver3(0,0,0,0,0,0.45,'b','LineWidth',1.8);
xlabel('X_c [m]'); ylabel('Y_c [m]'); zlabel('Z_c [m]');
title('Landmarks in the camera frame');
view(35,22);

%% Slide 20 — Camera intrinsics map normalized coordinates to pixels

image_width = 640;
image_height = 480;
fx = 800;
fy = 760;
skew = 0;
cx = 320;
cy = 240;

K = [fx skew cx; 0 fy cy; 0 0 1];
normalized_points = Pc ./ Pc(3,:);
pixel_h = K * normalized_points;
pixels_stepwise = pixel_h(1:2,:) ./ pixel_h(3,:);

fprintf('\nINTRINSIC PROJECTION\n');
fprintf('K:\n');
disp(K);
fprintf('Normalized coordinates [x_n; y_n]:\n');
disp(normalized_points(1:2,:));
fprintf('Pixel coordinates [u; v]:\n');
disp(pixels_stepwise);

nexttile;
hold on; grid on; axis equal;
plot(normalized_points(1,:),normalized_points(2,:),'o', ...
    'MarkerFaceColor',[0.00 0.55 0.80],'MarkerSize',7);
plot(0,0,'k+','MarkerSize',12,'LineWidth',2);
set(gca,'YDir','reverse');
xlabel('x_n = X_c/Z_c'); ylabel('y_n = Y_c/Z_c');
title('Normalized camera coordinates');
legend('Landmarks','Optical axis','Location','best','FontSize',8);

%% Slide 21 — One camera matrix maps a world point to a pixel

P = K * [R_cw t_cw];
pixels_matrix_h = P * [Pw; ones(1,size(Pw,2))];
pixels_matrix = pixels_matrix_h(1:2,:) ./ pixels_matrix_h(3,:);
pipeline_difference = vecnorm(pixels_stepwise - pixels_matrix,2,1);

selected_pixel = pixels_matrix(:,1);
ray_c = K \ [selected_pixel; 1];
ray_c = ray_c / norm(ray_c);
ray_w = R_wc * ray_c;
ray_depths = [1.5 3.0 5.0];
points_on_ray = Cw + ray_w * ray_depths;
ray_pixels_h = P * [points_on_ray; ones(1,numel(ray_depths))];
ray_pixels = ray_pixels_h(1:2,:) ./ ray_pixels_h(3,:);

fprintf('\nCOMPLETE CAMERA MATRIX\n');
fprintf('P = K[R|t]:\n');
disp(P);
fprintf('Maximum difference between stepwise and matrix projections: %.3e px\n', ...
    max(pipeline_difference));
fprintf('Different points on one world ray [m]:\n');
disp(points_on_ray);
fprintf('Their projected pixels:\n');
disp(ray_pixels);

nexttile;
hold on; grid on; axis equal;
rectangle('Position',[0 0 image_width image_height],'EdgeColor','k','LineWidth',1.4);
plot(pixels_stepwise(1,:),pixels_stepwise(2,:),'o','MarkerFaceColor',[0.95 0.45 0.10], ...
    'MarkerSize',7,'DisplayName','Projected landmarks');
plot(cx,cy,'k+','MarkerSize',12,'LineWidth',2,'DisplayName','Principal point');
set(gca,'YDir','reverse');
xlim([-20 image_width+20]); ylim([-20 image_height+20]);
xlabel('u [px]'); ylabel('v [px]');
title('Pixels after K[R|t]');
legend('Location','best','FontSize',8);

figure('Color','w','Name','Slide 21: pixel back-projection');
hold on; grid on; axis equal;
plot3(Cw(1),Cw(2),Cw(3),'kp','MarkerFaceColor','k','MarkerSize',12);
plot3(points_on_ray(1,:),points_on_ray(2,:),points_on_ray(3,:),'o', ...
    'MarkerFaceColor',[0.95 0.45 0.10],'MarkerSize',7);
plot3([Cw(1) points_on_ray(1,end)],[Cw(2) points_on_ray(2,end)], ...
    [Cw(3) points_on_ray(3,end)],'-','Color',[0.00 0.45 0.70],'LineWidth',2);
xlabel('X_w [m]'); ylabel('Y_w [m]'); zlabel('Z_w [m]');
title('One pixel back-projects to a 3D ray');
legend('Camera center','Possible 3D points','Viewing ray','Location','best');
view(35,22);

%% Slide 22 — Numerical projection from the presentation

K_example = [800 0 320; 0 800 240; 0 0 1];
Pc_near = [0.10; 0.05; 2.0];
Pc_far = [0.10; 0.05; 4.0];

pixel_near_h = K_example * Pc_near;
pixel_near = pixel_near_h(1:2) / pixel_near_h(3);
pixel_far_h = K_example * Pc_far;
pixel_far = pixel_far_h(1:2) / pixel_far_h(3);
principal_point = K_example(1:2,3);
distance_near = norm(pixel_near-principal_point);
distance_far = norm(pixel_far-principal_point);

fprintf('\nNUMERICAL DEPTH EXAMPLE\n');
fprintf('P_c = [%.2f %.2f %.2f]^T m -> pixel (%.1f, %.1f)\n', ...
    Pc_near(1),Pc_near(2),Pc_near(3),pixel_near(1),pixel_near(2));
fprintf('P_c = [%.2f %.2f %.2f]^T m -> pixel (%.1f, %.1f)\n', ...
    Pc_far(1),Pc_far(2),Pc_far(3),pixel_far(1),pixel_far(2));
fprintf('Distance-to-principal-point ratio after doubling depth: %.3f\n', ...
    distance_far/distance_near);

figure('Color','w','Name','Slide 22: numerical depth example');
plot(principal_point(1),principal_point(2),'k+','MarkerSize',14,'LineWidth',2); hold on;
plot(pixel_near(1),pixel_near(2),'o','MarkerSize',9,'MarkerFaceColor',[0.95 0.45 0.10]);
plot(pixel_far(1),pixel_far(2),'o','MarkerSize',9,'MarkerFaceColor',[0.00 0.55 0.80]);
plot([principal_point(1) pixel_near(1)],[principal_point(2) pixel_near(2)],'k--');
axis equal; grid on; xlim([300 380]); ylim([225 275]); set(gca,'YDir','reverse');
xlabel('u [px]'); ylabel('v [px]');
title('Increasing depth moves the pixel toward the principal point');
legend('Principal point','Z_c = 2 m','Z_c = 4 m','Location','best');
