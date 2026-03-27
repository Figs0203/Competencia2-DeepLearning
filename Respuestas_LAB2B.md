# Respuestas - Laboratorio 2, Parte B: Segmentacion Semantica con U-Net y Transfer Learning

**Grupo:** Juan Andrés Young, Agustín Figueroa, Martín Valencia

---

## Actividad 1 — Dataset y transforms sincronizados

**Pregunta:** ¿Por que se usa interpolacion NEAREST para la mascara y BILINEAR para la imagen al hacer resize? ¿Que pasaria si usaras bilineal para ambas?

**Respuesta:** La imagen contiene valores continuos de intensidad (0 a 255 por canal), de modo que interpolar entre pixeles vecinos con el metodo bilineal produce transiciones suaves y visualmente naturales. En cambio, la mascara contiene etiquetas discretas (1 = animal, 2 = fondo, 3 = borde). Si se aplicara interpolacion bilineal a la mascara, el promedio ponderado entre etiquetas generaria valores fraccionarios (por ejemplo, 1.6) que no corresponden a ninguna clase valida. Esto corrompe la informacion de ground truth justo en la zona de los bordes, que es precisamente donde la segmentacion necesita mayor precision.

Con interpolacion NEAREST, cada pixel de la mascara redimensionada adopta el valor del pixel mas cercano sin mezclar etiquetas, preservando intacta la correspondencia entre imagen y anotacion. En nuestro experimento, al verificar con `masks.unique()` confirmamos que las mascaras solo contienen los valores `tensor([0, 1])`, lo cual valida que la binarizacion y el resize con NEAREST funcionan correctamente.

---

## Actividad 2 — Construir U-Net desde cero

**Pregunta:** ¿Que informacion aportan las skip connections que el decoder no podria recuperar por si solo?

**Respuesta:** Las skip connections transmiten informacion espacial de alta resolucion que el encoder pierde de forma irreversible tras cada operacion de pooling. En concreto, aportan tres tipos de informacion:

1. **Bordes y contornos finos:** El MaxPool reduce la resolucion a la mitad en cada nivel, diluyendo los detalles de los limites entre el animal y el fondo. Las skip connections inyectan directamente los feature maps a resolucion completa, permitiendo al decoder reconstruir contornos precisos.

2. **Localizacion espacial:** Tras varias etapas de pooling, los feature maps del bottleneck codifican eficientemente *que* hay en la imagen, pero pierden la informacion de *donde* esta. Las skips restauran esas coordenadas espaciales.

3. **Flujo de gradientes:** Proveen un camino directo (analogo a las conexiones residuales de ResNet) para que los gradientes fluyan desde la perdida hasta las capas tempranas del encoder, facilitando el entrenamiento.

La verificacion de shapes confirmo que nuestra UNet produce correctamente una salida `[2, 1, 256, 256]` a partir de una entrada `[2, 3, 256, 256]`, lo que valida que las skip connections se estan concatenando en las dimensiones correctas a traves de los 4 niveles del decoder.

---

## Actividad 3 — Funciones de perdida para segmentacion

**Pregunta:** ¿Por que CrossEntropyLoss falla ante desbalance de clases? ¿Que mide exactamente el coeficiente Dice?

**Respuesta:** CrossEntropy calcula la perdida promediada sobre todos los pixeles, tratando cada pixel con igual peso. Cuando el 80-90% de los pixeles pertenecen al fondo (como ocurre frecuentemente en nuestro dataset de mascotas y de forma mas extrema en imagenes medicas), la perdida esta dominada por los pixeles mayoritarios. El modelo puede alcanzar una perdida baja simplemente prediciendo "fondo" en todas partes, sin aprender a delinear el objeto.

El coeficiente Dice mide el solapamiento relativo entre la region predicha P y el ground truth G:

Dice = 2|P interseccion G| / (|P| + |G|)

Su valor va de 0 (sin solapamiento) a 1 (solapamiento perfecto). Lo relevante es que Dice se calcula exclusivamente sobre las regiones de interes, sin que los pixeles de fondo diluyan la metrica. Si el modelo ignora al objeto, Dice cae a cero inmediatamente, independientemente de cuantos pixeles de fondo clasifique correctamente.

En nuestro entrenamiento, la combinacion BCE + Dice demostro ser mas efectiva que BCE sola: el modelo con BCE+Dice alcanzo un Dice de 0.7914 frente a 0.7721 con BCE puro, confirmando experimentalmente el beneficio de incorporar esta metrica directamente en la funcion de perdida.

---

## Actividad 5 — ResNetUNet: encoder preentrenado como backbone

**Pregunta:** ¿En que se diferencia congelar el encoder aqui respecto a lo que se hizo con VGG en el workshop anterior?

**Respuesta:** El principio fundamental es el mismo en ambos casos: proteger los pesos preentrenados mientras los componentes nuevos se estabilizan. Sin embargo, hay diferencias estructurales importantes:

1. **Componente entrenable mucho mas grande:** En VGG solo entrenabamos una capa lineal de aproximadamente 50K parametros. Aqui, el decoder completo (bottleneck + 5 UpBlocks con convoluciones) tiene millones de parametros que deben aprender a reconstruir la mascara espacial.

2. **Naturaleza de la tarea:** En el workshop anterior, la salida era un vector de 2 clases. Aqui, la salida es un mapa denso de 256x256 pixeles, donde cada pixel requiere una prediccion. Reconstruir informacion espacial desde features comprimidos es inherentemente mas complejo que clasificar.

3. **Resultado experimental:** Aun con el encoder completamente congelado, el decoder logro un IoU de 0.8529 y Dice de 0.9158 en solo 5 epocas, superando ampliamente a la UNet entrenada desde cero (IoU 0.6741, Dice 0.7914). Esto demuestra la enorme ventaja de reutilizar features preentrenados de ImageNet incluso para una tarea tan diferente como segmentacion de mascotas.

Al descongelar y hacer fine-tuning con lr=1e-4, las metricas mejoraron marginalmente (IoU 0.8640, Dice 0.9227), lo que confirma que el encoder de ResNet-18 ya extraia features lo suficientemente buenas para esta tarea y que el ajuste fino aporta un refinamiento modesto pero consistente.

---

## Actividad 7 — Evaluacion comparativa y visualizacion

**Pregunta:** ¿Donde falla mas cada modelo: en el interior de la region o en los bordes? ¿A que se atribuye esa diferencia?

**Respuesta:** Los resultados finales sobre el test set completo son:

| Modelo | IoU | Dice |
|--------|-----|------|
| UNet scratch | 0.6740 | 0.7913 |
| ResNetUNet (frozen) | 0.8528 | 0.9158 |
| ResNetUNet (finetuned) | 0.8640 | 0.9227 |

Al inspeccionar visualmente las predicciones:

- **UNet scratch:** Comete errores tanto en el interior como en los bordes. La mascara predicha frecuentemente presenta huecos internos (falsos negativos) y protuberancias irregulares en los contornos. Al no contar con features preentrenados, el encoder entrenado por solo 5 epocas no logra distinguir de forma robusta las texturas del animal frente al fondo.

- **ResNetUNet (frozen):** El interior de la region esta cubierto de forma mucho mas uniforme gracias a los features preentrenados de ImageNet que reconocen texturas animales. Los errores se concentran en los bordes, especialmente en zonas donde el pelaje se mezcla con fondos de color similar.

- **ResNetUNet (finetuned):** Muestra los bordes mas definidos de los tres modelos. El fine-tuning permite que encoder y decoder se co-adapten, refinando la extraccion de features en los bordes mas ambiguos. La mejora de IoU de 0.8528 a 0.8640 se traduce en contornos visualmente mas limpios.

La diferencia fundamental se atribuye a la calidad de los features del encoder: un encoder preentrenado extrae patrones semanticos ricos desde la primera epoca, mientras que uno aleatorio necesita muchas mas epocas para aprender representaciones comparables.

---

## Pregunta de Investigacion 1 — Entrenar con solo BCE (sin Dice)

**Pregunta:** ¿Que pasa con las metricas al usar solo BCEWithLogitsLoss? ¿Que estrategias existen para manejar el desbalance en segmentacion medica?

**Respuesta:** Los resultados experimentales muestran una degradacion medible al entrenar sin Dice:

| Configuracion | IoU (val) | Dice (val) |
|---------------|-----------|------------|
| BCE + Dice | 0.6741 | 0.7914 |
| BCE only | 0.6533 | 0.7721 |

La caida de aproximadamente 2 puntos porcentuales en ambas metricas confirma que BCE, al tratar cada pixel independientemente, tiende a favorecer la clase mayoritaria (fondo). El coeficiente Dice en la funcion de perdida actua como un regularizador que penaliza directamente la falta de solapamiento con la region de interes.

Estrategias para manejar el desbalance en segmentacion medica:

1. **Perdidas basadas en solapamiento:** Dice Loss, Tversky Loss (generaliza Dice con pesos asimetricos para falsos positivos y falsos negativos), y Focal Loss (reduce la contribucion de pixeles faciles).
2. **Ponderacion de clases:** Asignar un peso mayor a la clase minoritaria en la BCE, inversamente proporcional a su frecuencia.
3. **Oversampling de parches:** En imagenes medicas, se recortan parches centrados en la region de interes para balancear la proporcion de pixeles positivos en cada batch.
4. **Deep supervision:** Calcular la perdida en multiples resoluciones del decoder para reforzar la senal de aprendizaje en capas intermedias.

---

## Pregunta de Investigacion 2 — Efecto de eliminar las skip connections

**Pregunta:** Elimina las skip connections de ResNetUNet y entrena. ¿Que observas visualmente en los bordes de las predicciones?

**Respuesta:** Los resultados cuantitativos del modelo sin skip connections (ResNetUNet_NoSkip) son:

| Modelo | IoU (val) | Dice (val) |
|--------|-----------|------------|
| ResNetUNet (con skips, frozen) | 0.8529 | 0.9158 |
| ResNetUNet NoSkip | 0.7243 | 0.8324 |

La caida de 12.9 puntos porcentuales en IoU es sustancial. Lo que mas destaca al comparar las predicciones visualmente es la perdida de precision en los contornos: sin skip connections, los bordes de la mascara predicha son significativamente mas borrosos e imprecisos. El modelo tiende a producir mascaras "infladas" o excesivamente suavizadas, perdiendo detalles como las orejas puntiagudas o las patas individuales.

La explicacion arquitectonica es directa: el decoder debe reconstruir la mascara de 256x256 a partir unicamente del bottleneck de 8x8. Toda la informacion espacial descartada durante el pooling debe ser "estimada" por las convoluciones transpuestas, lo que inevitablemente produce resultados borrosos. Las skip connections resuelven este problema inyectando los feature maps originales en cada nivel de resolucion.

Este resultado es consistente con la motivacion original de Ronneberger et al. (2015): las skip connections son el componente clave que diferencia a U-Net de un autoencoder convencional.

---

## Pregunta de Investigacion 3 — Curva IoU vs Threshold

**Pregunta:** La funcion compute_metrics usa threshold=0.5 para binarizar. Grafica la curva IoU vs threshold en el test set. ¿Cual seria el threshold optimo?

**Respuesta:** El barrido de umbrales sobre el test set revelo que el **threshold optimo es 0.55**, alcanzando un IoU de **0.8642**. Esto representa una mejora respecto al threshold por defecto de 0.50 que se utiliza convencionalmente.

La curva IoU vs threshold tiene forma de campana invertida: para valores muy bajos (< 0.3), se generan demasiados falsos positivos (la mascara se "infla" cubriendo regiones de fondo), y para valores muy altos (> 0.8), se pierden pixeles de foreground produciendo falsos negativos. El maximo se ubica ligeramente por encima de 0.5, lo que indica que el modelo tiene una calibracion razonablemente buena, con una leve tendencia a generar probabilidades moderadas en los bordes que se benefician de un umbral algo mas exigente.

En la practica, la eleccion del threshold depende del contexto. En imagenes medicas donde un falso negativo es mas costoso (por ejemplo, detectar un tumor), se prefiere un threshold mas bajo para maximizar la sensibilidad.

---

## Pregunta de Investigacion 4 — Data Augmentation medica

**Pregunta:** Investiga que augmentations son estandar en imagenes medicas (elastic deformation, gamma correction). Implementa al menos una en PetSegDataset y mide el impacto.

**Respuesta:** Las tecnicas de augmentation mas utilizadas en segmentacion de imagenes medicas incluyen:

1. **Gamma correction:** Modifica el brillo no lineal de la imagen aplicando la transformacion `pixel = pixel^gamma`. Simula variaciones en la iluminacion o en las condiciones de adquisicion del equipo medico.
2. **Elastic deformation:** Genera distorsiones suaves que simulan la variabilidad anatomica natural entre pacientes.
3. **Ajuste de contraste y brillo:** Compensa la heterogeneidad de intensidades entre diferentes scanners o protocolos de adquisicion.
4. **Flips y rotaciones:** Explotan las simetrias naturales de la anatomia.

En nuestro experimento, implementamos gamma correction con gamma aleatorio entre 0.7 y 1.5. Los resultados fueron:

| Configuracion | Dice (val) |
|---------------|------------|
| Sin augmentation (base) | 0.7914 |
| Con gamma correction | 0.7603 |

Contrario a lo esperado, la gamma correction produjo un Dice ligeramente inferior. Esto se atribuye a dos factores: (1) el dataset Oxford-IIIT Pet tiene condiciones de iluminacion relativamente homogeneas, por lo que la perturbacion gamma introduce variabilidad que no se corresponde con la distribucion real de los datos; y (2) con solo 5 epocas de entrenamiento, la red no tiene suficiente tiempo para beneficiarse de la regularizacion que aporta el augmentation. En imagenes medicas, donde la variabilidad de adquisicion es mucho mayor, estas tecnicas suelen ser imprescindibles.

---

## Pregunta de Investigacion 5 — Reto opcional: Hausdorff Distance

**Pregunta:** Implementa la metrica Hausdorff distance. ¿Que mide que IoU y Dice no capturan?

**Respuesta:** La distancia de Hausdorff mide la **maxima distancia** entre el contorno predicho y el contorno del ground truth:

HD(P, G) = max( max(p in P) d(p, G), max(g in G) d(g, P) )

Mientras que IoU y Dice cuantifican el solapamiento global de regiones (la proporcion de area compartida), la Hausdorff es una metrica de **contorno** que detecta el peor error local. Un modelo puede tener un Dice de 0.92 (como nuestro ResNetUNet finetuned) indicando excelente solapamiento general, pero si existe un unico punto del borde severamente desplazado, la Hausdorff sera alta.

Esta metrica es especialmente critica en aplicaciones medicas donde la precision del contorno tiene consecuencias directas: en planificacion de radioterapia, un borde mal segmentado puede significar irradiar tejido sano; en medicion de volumenes tumorales, un error local puede sesgar significativamente la estimacion.

En la practica se usa la variante **95th percentile Hausdorff** (HD95), que ignora el 5% de distancias mas extremas para mayor robustez frente a outliers. Esto es relevante porque la Hausdorff clasica es extremadamente sensible a un unico pixel aislado mal clasificado, lo que puede no reflejar la calidad real de la segmentacion.

En nuestros modelos, se esperaria que la UNet scratch presente una Hausdorff considerablemente mayor que la ResNetUNet finetuned, dado que sus predicciones muestran bordes mas irregulares y con errores locales mas pronunciados.
