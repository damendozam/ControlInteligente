%% 
% Glosario de variables:

% Q -> Número de ejemplares
% K -> Número de clases o categorías
% a -> Factor de la exponencial (decay rate)
% eta -> Tasa (velocidad) de aprendizaje (step gain > 0)
% I -> Número de épocas (epochs) para el entrenamiento
% ------------- "capa" de entrada --------------
% Xn -> Vector de características, valores de entrada a la red
% N -> Número de características, nodos a la entrada
% ---------- capa oculta o intermedia ----------
% Wnm -> Pesos de conexiones entre nodos de entrada y capa oculta
% Rm -> Vector con sumas ponderadas de cada nodo en la capa oculta
% b1 -> Bias (treshhold), umbrales en la capa oculta
% h -> Función de activación de los nodos en la capa oculta
% Ym -> Vector con salidas de la capa oculta o intermedia
% M -> Número de neurodos en la capa oculta o intermedia
% --------------- capa de salida ---------------
% Umj -> Pesos de conexiones entre capa oculta y capa de salida
% Sj -> Vector con sumas ponderadas de cada nodo en la capa de salida
% b2 -> Bias (treshhold), umbrales en la capa de salida
% g -> Función de activación de los nodos en la capa de salida
% Zj -> Vector con salidas de la red neuronal
% J -> Número de neurodos en la capa de salida
% ----------------------------------------------
% Tj -> Vector con valores objetivo (target)
% E -> Error de suma cuadrática total (TSSE)
% TSSE -> Error medio cuadrático total: TMSE = E/(Q*J)
%% 
% Lectura de datos:

clear all, close all, clc

% -------------- Inputs ----------------
% % Lectura del archivo de datos:
prompt = 'Ingrese el nombre del archivo [Dataset.xlsx]: ';
str = input(prompt,'s');
if isempty(str)
    str = 'Dataset.xlsx';
end
datosTab = readtable(str);
%% 
% Definición de hiperparámetros:

% El usuario define el número de neurodos
M = input('Número de neurodos en la capa oculta: ')+1;
% Número de épocas
I = 5000;
% Taza de aprendizaje (valor recomendado eta=.5)
eta = .5;

% Tasa de decaimiento (función de activación)
a = 1;
% Valores de bias en capa oculta y de salida
b1 = 1;
b2 = 1;

% Gráfica de la función de activación implementada:
figure
t = -8:.5:8;
f =  1./(1+exp(-a*t+b1));
plot(t,f,'Color',[0.5 0 0])
grid on
%% 
% Tratamiento inicial de datos e inicialización de la red neuronal:

% Convertir de tabla a matriz numérica:
D = table2array(datosTab);
% Gráfica del dataset:
figure
scatter(D(:,1),D(:,2),25,D(:,3),'filled')
grid on

% Extraer número total de ejemplares:
Q = size(D,1)
% 70% de los datos para entrenamiento:
Q_train = .7 * Q
% 30% restante de los datos para validación:
Q_validate = Q - Q_train
% División aleatoria del dataset en datos de entrenamiento y validación:
k = randperm(Q, Q_train);
D_train  = D(k, :);
r = true(1,Q);
r(k) = false;
D_validate = D(r, :);

% Definir vector de entrada y vector objetivo (entrenamiento)
X = [D_train(:,[1 2]);[1 1]]';
T = [D_train(:,[3 4]);ones(2)]';
% Extraer # de nodos en capa de entrada
N = size(X,1);
% Extraer # de neurodos en capa de salida
J = size(T,1);

% Definir vector de entrada y vector objetivo (validación)
X_v = D_validate(:,1:N)';
T_v = [D_validate(:,N+1:N+J);[1 1]]';

% ---------- capa oculta o intermedia ----------
% Definir matriz de pesos con valores aleatorios entre -.5 y .5
W = -.5 + (.5-(-.5)).*rand(N,M) % igual a W=rand(N,M)-.5
% -------------- capa de salida ----------------
% Definir matriz de pesos con valores aleatorios entre -.5 y .5
U = -.5 + (.5-(-.5)).*rand(M,J) % igual a U=rand(M,J)-.5
%% 
% Algoritmo de retro-propagación del error (Error Backpropagation):

% Asignación espacios de memoria para las matrices (preallocation):
evolW = zeros(N,M,I);
evolU = zeros(M,J,I);
TMSE_train = zeros(I,1);
TMSE_validate = zeros(I,1);
% Entrenamiento y validación por épocas:
for r = 1:I
    % --------------------------------------------
    % Entrenamiento de la red neuronal:
    for q = 1:Q_train
        % "UpdateNN()" -> Función para actualizar la red neuronal
        [Z,Y] = actualizarNN(X,W,U,a); 
        for m = 1:M
            % Descenso del Gradiente
            error = 0;
            for j = 1:J
                U(m,j) = U(m,j) + eta*(T(j,q)-Z(j,q))*Z(j,q)*(1-Z(j,q))*Y(m,q);
                error = error + (T(j,q)-Z(j,q));
            end
            for n = 1:N
                Wdelta = eta*error*Z(j,q)*(1-Z(j,q))*U(m,j)*Y(m,q)*(1-Y(m,q))*X(n,q);
                W(n,m) = W(n,m) + Wdelta;
            end
        end
    end
    % Error de suma cuadrática total (TSSE) de entrenamiento:
    E_train = sum(sum((T-Z).^2));
    % Error medio cuadrático total (TMSE) de entrenamiento:
    TMSE_train(r,:) = E_train/(J*Q_train);
    % Vectores de evolución de pesos:
    evolU(:,:,r) = U;
    evolW(:,:,r) = W;
    % --------------------------------------------
    % Validación de la red neuronal:
    for q = 1:Q_validate
        % Actualizar la red neuronal:
        [Z_v,Y_v] = actualizarNN(X_v,W,U,a);
    end
     % Error de suma cuadrática total (TSSE) de validación:
     E_validate = sum(sum((T_v-Z_v).^2)); 
     % Error medio cuadrático total (TMSE) de validación:
    TMSE_validate(r,:) = E_validate/(J*Q_validate);
end

% --------------------------------------------
% Gráfica de error medio cuadrático
r = 1:1:I;
figure
hold on
% Gráfica del TMSE para entrenamiento:
plot(r,TMSE_train,'Color',[.5 0 0])
% Gráfica del TMSE para validación:
plot(r,TMSE_train,'Color',[0 .5 0])
grid on
legend('Entrenamiento','Validación')

% --------------------------------------------
% Gráficas de la evolución de pesos
r = 1:1:I;
% Pesos hacia capa oculta:
figure
t = tiledlayout(M,N); % para graficar en una sola figura
for m = 1:M
    for n = 1:N
        nexttile
        plot(r, squeeze(evolW(n,m,:)),'Color',[0 0 .5]);
        title(strcat('W','_',num2str(n),'_',num2str(m) ));
        grid on
    end
end
t.Padding='compact'; t.TileSpacing='compact';
% Pesos hacia capa de salida:
figure
t = tiledlayout(M,J); % para graficar en una sola figura
for m = 1:M
    for j = 1:J
        nexttile
        plot(r, squeeze(evolU(m,j,:)),'Color',[.5 0 .5]);
        title(strcat('U','_',num2str(m),'_',num2str(j) ));
        grid on
    end
end
t.Padding='compact'; t.TileSpacing='compact';

% --------------------------------------------
% Gráficas de separación de clases
% Entrenamiento:
graficar(X,T,'Entrenemiento - Objetivo');
graficar(X,Z,'Entrenemiento - Resultado');
% Validación:
graficar(X_v,T_v,'Validación - Objetivo');
graficar(X_v,Z_v,'Validación - Resultado');
%% 
% Funciones definidas:

function [Z,Y] = actualizarNN(X,W,U,a)
    % Sumatoria ponderada en cada neurodo:
    R = W'*X;
    [mU,nU]=size(U);
    % Función de activación unipolar Y = h(R)
    Y = [1./(1+exp(-a*R+W(length(W)))) ones(mU,1)];
    % Sumatoria ponderada en cada neurodo:
    S = U'*Y;
    % Función de activación unipolar Z = g(S)
    Z = 1./(1+exp(-a*S+U(length(U))));
end

% Función para graficar separación de clases:
function graficar(X,Z,titulo)
    x = [X(1,:) 1]';
    y = [X(2,:) 1]';
    z = Z(1,:)';
    size(x)
    size(y)
    size(z)
    xg = linspace(min(x), max(x), 100);
    yg = linspace(min(y), max(y), 100);
    figure;
    [xg,yg] = meshgrid(xg,yg);
    F = scatteredInterpolant([x y],z,'natural','none');
    zg = F(xg,yg);
    hp = pcolor(xg,yg,zg);
    shading flat;
    hold on;
    hs = scatter(x,y,25,'k','filled');
    colorbar;
    title(titulo);
end

