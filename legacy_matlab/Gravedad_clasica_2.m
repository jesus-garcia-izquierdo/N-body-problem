function [x,y,z,vx,vy,vz,t,px,py,pz,p,L,Ecin,Epot,Etot] = Gravedad_clasica_2(m,x0,y0,z0,vx0,vy0,vz0,tf,h)

% Notación de los vectores: Filas indican el cuerpo y las columnas, el instante

np=tf/h; %Número de puntos
t(1)=0;
x=x0;
y=y0;
z=z0;
vx=vx0;
vy=vy0;
vz=vz0;


for i=[1:np] % Recorre todos los instantes

    t(i+1)=t(i)+h;

    for j=[1:length(m)] %Recorre todos los planetas

        delta_x=0;
        delta_y=0;
        delta_z=0;

        for k=[1:length(m)] %Para calcular la distancia de un planeta a todos

            %Calculamos la distancia entre los dos cuerpos
            %delta x se calcula para tener la aceleracion (distancia entre cuerpos)
            %x(j,i)-x(k,i) significa restar la distancia entre el
            %primer cuerpo con el segundo cuerpo, en el instante i

            if k==1
                delta_x=x(j,i)-x(k,i);
                delta_y=y(j,i)-y(k,i);
                delta_z=z(j,i)-z(k,i);
            else
                delta_x(end+1)=x(j,i)-x(k,i);
                delta_y(end+1)=y(j,i)-y(k,i);
                delta_z(end+1)=z(j,i)-z(k,i);
            end
        end

        delta_x=delta_x';%Para que el vector sea un vector columna, podía haber cambiado el vector masa
        delta_y=delta_y';
        delta_z=delta_z';


        %Calculamos la aceleracion para 1 cuerpo producida por todos los cuerpos

        [ax1,ay1,az1]=a(m,delta_x,delta_y,delta_z);

        %Calculamos la posicion de un cuerpo en ese instante

        x1(j,1)=x(j,i)+vx(j,i)*h+sum(ax1)*h*h/2;%Hacemos la suma por superposicion de aceleraciones
        y1(j,1)=y(j,i)+vy(j,i)*h+sum(ay1)*h*h/2;
        z1(j,1)=z(j,i)+vz(j,i)*h+sum(az1)*h*h/2;

        %Los ponemos como vectores filas porque luego para calcular la velocidad
        %tenemos que utilizar las aceleraciones

        ax(j,1)=sum(ax1);
        ay(j,1)=sum(ay1);
        az(j,1)=sum(az1);
        %Guardamos las aceleraciones

    end

    %Introduzco los valores en x,y,z
    x=[x x1];
    y=[y y1];
    z=[z z1];

    for j=[1:length(m)]%Pasa por todos los planetas


        delta_x=0;
        delta_y=0;
        delta_z=0;


        for k=[1:length(m)]%Para calcular la distancia del un planeta a todos


            %Calculamos la distancia entre los dos cuerpos
            %delta x se calcula para tener la aceleracion(distancia entre cuerpos)
            %x(j,i)-x(k,i) significa restar la distancia entre el
            %primer cuerpo con el segundo cuerpo, en el instante i

            if k==1
                delta_x=x(j,i+1)-x(k,i+1);
                delta_y=y(j,i+1)-y(k,i+1);
                delta_z=z(j,i+1)-z(k,i+1);
            else
                delta_x(end+1)=x(j,i+1)-x(k,i+1);
                delta_y(end+1)=y(j,i+1)-y(k,i+1);
                delta_z(end+1)=z(j,i+1)-z(k,i+1);
            end
        end

        delta_x=delta_x';%Para que el vector sea un vector columna
        delta_y=delta_y';
        delta_z=delta_z';


        %Calculamos la aceleracion para 1 cuerpo producida por
        %todos los cuerpos

        [ax2,ay2,az2]=a(m,delta_x,delta_y,delta_z);

        %Calculamos la velocidad de un cuerpo
        vx1(j,1)=vx(j,i)+(ax(j)+sum(ax2))*h/2;
        vy1(j,1)=vy(j,i)+(ay(j)+sum(ay2))*h/2;
        vz1(j,1)=vz(j,i)+(az(j)+sum(az2))*h/2;
        %El 1 es por cambiar el nombre


    end

    vx=[vx vx1];
    vy=[vy vy1];
    vz=[vz vz1];

end


%Momento lineal
px=m'*vx;
py=m'*vy;
pz=m'*vz;

p=sqrt(px.*px+py.*py+pz.*pz);

%Energía cinetica
v=sqrt(vx.*vx+vy.*vy+vz.*vz);
Ecin=1/2*m'*(v.*v);

%Energía potencial

for i=1:length(m)%Para todos los planetas

    for j=1:length(m)%Para calcular la energia potencial de un planeta a los otros

        %Calculamos la distancia entre los dos cuerpos
        %delta x se calcula para tener la energía potencial(distancia entre cuerpos)
        %x(j,i)-x(k,i) significa restar la distancia entre el
        %primer cuerpo con el segundo cuerpo, en el instante i
        delta_x=x(i,:)-x(j,:);
        delta_y=y(i,:)-y(j,:);
        delta_z=z(i,:)-z(j,:);

        modulo=sqrt(delta_x.*delta_x+delta_y.*delta_y+delta_z.*delta_z);

        if j==1 && i==1

            Epot=Epot1(m(i),m(j),modulo);

        else

            Epot=[Epot; Epot1(m(i),m(j),modulo)];

        end


    end


end
Epot=sum(Epot,1)/2;


%Energía total
Etot=Ecin+Epot;


%Momento angular
for n=1:length(m)

    if n==1
        px=m(n)*vx(n,:);
        py=m(n)*vy(n,:);
        pz=m(n)*vz(n,:);

    else

        px=[px;m(n)*vx(n,:)];
        py=[py;m(n)*vy(n,:)];
        pz=[pz;m(n)*vz(n,:)];
    end

end

%Cálculo del momento angular (por determinantes)
Lx=y.*pz-z.*py;
Ly=z.*px-x.*pz;
Lz=x.*py-y.*px;

%El módulo del L(tot)= mod(Lx(tot)+Ly(tot)+Lz(tot))
Lx=sum(Lx,1);
Ly=sum(Ly,1);
Lz=sum(Lz,1);
L=sqrt(Lx.*Lx+Ly.*Ly+Lz.*Lz);

end


function [ax, ay, az]=a(M,delta_x,delta_y,delta_z)

% Función que devuelve la aceleración (en vectores fila) a partir de la posición.

G=6.67384e-11;
modulo=sqrt(delta_x.*delta_x+delta_y.*delta_y+delta_z.*delta_z).^3;
ax=(-G*M.*delta_x./modulo)';
ay=(-G*M.*delta_y./modulo)';
az=(-G*M.*delta_z./modulo);


% Hay posiciones en los que delta es igual a 0 (la posición de un planeta sobre sí mismo), por lo que,
% según la ecuación anterior, a tendería a infinito. Se buscan en los que% eso pasa, y se iguala a 0
ax(find(isnan(ax)))=0;
ay(find(isnan(ay)))=0;
az(find(isnan(az)))=0;

end

function [E]=Epot1(M1,M2,modulo)

% Función que devuelve la aceleración a partir de la posición.

G=6.67384e-11;
E=-G.*M1.*M2./modulo;

E(find(E==-Inf))=0; % Se hace lo mismo que con la función de aceleración

end
