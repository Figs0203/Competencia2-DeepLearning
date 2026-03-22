# Respuestas a Preguntas Conceptuales — Laboratorio 2 (Parte A)
**Grupo:** Juan Andrés Young, Agustín Figueroa, MartínValencia  
**Dataset:** Cats vs. Dogs (Kaggle)

---

## Actividad 1 — Dataset y pipeline de preprocesamiento

**Pregunta:** ¿De dónde vienen esos valores de media y std? ¿Por qué es importante usarlos si vas a cargar pesos preentrenados?
> **Respuesta:**
> - **Origen de mean/std de ImageNet:** Se calcularon por canal (R, G, B) sobre aproximadamente 1.2 millones de imágenes del training set original de ImageNet (ILSVRC 2012).
> - **Importancia:** Los pesos preentrenados (filtros convolucionales) asumen estrictamente esta distribución de entrada. Proveer valores con escalas diferentes produce activaciones numéricamente incorrectas y causa que los filtros no se activen como se espera, degradando fuertemente el desempeño del transfer learning.

---

## Actividad 2 — Modelos preentrenados y extracción de features

**Preguntas:** ¿Cuántos parámetros tiene VGG-16 en total? ¿Qué hace la capa Dropout y por qué mejora la generalización? ¿Por qué la última capa tiene 1000 salidas?
> **Respuesta:**
> - **Parámetros VGG-16:** Tiene ~138 millones de parámetros en total (14.7M en las capas convolucionales y 123.6M en las capas Full-Connected).
> - **Dropout:** Durante el entrenamiento, desactiva aleatoriamente una fracción (usualmente el 50%) de las neuronas en esa capa. Esto obliga a la red a no depender de unas pocas neuronas específicas, forzando representaciones robustas y distribuidas (actuando como un ensamble implícito para evitar sobreajuste).
> - **1000 salidas:** El dataset original sobre el cual fue entrenada la red VGG-16 (ImageNet ILSVRC) tiene configuradas exactamente 1000 categorías distintas para clasificar.

---

## Actividad 3 — Clasificador lineal sobre features precomputadas

**Preguntas:** ¿Cuál es la diferencia entre CrossEntropyLoss y NLLLoss + LogSoftmax? ¿Por qué combinarlos es un error?
> **Respuesta:**
> - **Diferencia:** Son matemáticamente equivalentes al final del flujo, pero `CrossEntropyLoss` combina `LogSoftmax` y `NLLLoss` internamente usando un truco matemático (Log-Sum-Exp) que lo hace muchísimo más estable numéricamente (evita underflow/overflow con valores extremos).
> - **Error de combinarlos:** Al usar una red que terminaliza en un `LogSoftmax`, las salidas ya están pasadas por la activación softmax. Si luego se le pasa esto a una función de pérdida `CrossEntropyLoss` (que también aplica internamente un softmax sobre sus entradas), se produce una doble aplicación de la operación de probabilidad. Esto aplana irremediablemente la distribución, lo que atenúa los gradientes y destruye la convergencia del entrenamiento.

---

## Actividad 4 — Transfer Learning: reemplazar clasificador y congelar

**Pregunta:** ¿Por qué es importante congelar las capas convolucionales antes de entrenar el nuevo clasificador?
> **Respuesta:**
> Al reemplazar el clasificador por uno nuevo (de 2 clases), sus pesos se inicializan aleatoriamente. Si hiciéramos *backpropagation* desde el principio por toda la red, los grandes errores iniciales del clasificador aleatorio propagarían gradientes enormes hacia atrás, destruyendo los valiosos filtros preentrenados de las convoluciones. Al congelar las capas, estabilizamos primero el nuevo clasificador y protegemos la base de características. Una vez estable, recién ahí se realiza un "fine-tuning" descongelando con un learning rate (tasa de aprendizaje) mucho menor.

---

## Actividad 6 — Adaptar ResNet-18

**Preguntas:** ¿Qué es un residual block? ¿Qué problema resuelve en redes muy profundas? ¿ResNet-18 entrena más rápido que VGG? ¿Por qué?
> **Respuesta:**
> - **Residual block:** Introduce una conexión directa (skip connection/identity shortcut) que suma la entrada del bloque directamente a la salida de las convoluciones subyacentes (`y = F(x) + x`). Las capas interiores entonces sólo necesitan aprender "correcciones" o "residuales" en lugar de transformaciones completas.
> - **Problema que resuelve:** El desvanecimiento del gradiente (vanishing gradient problem) presente en redes muy profundas. Al proveer un "atajo" directo, los gradientes fluyen inalterados hacia las primeras capas, permitiendo entrenar redes de cientos de capas con facilidad.
> - **Velocidad VGG vs ResNet-18:** Sí, ResNet-18 entrena sustancialmente más rápido. VGG-16 es una arquitectura redundante con casi 138 millones de parámetros (dominados por gigantescas capas Full-Connected al final), mientras que ResNet-18 aprovecha su diseño de bloques y Average Pooling final para tener solo ~11.7M de parámetros, haciéndola computacionalmente más ligera, sumado a que usa "BatchNorm" para converger en muchas menos épocas.

---

## Preguntas de Investigación (Sección 10)

**P1. ¿Cuándo NO conviene usar transfer learning? Da al menos dos escenarios concretos.**
> **Respuesta:**
> 1. **Dominio radicalmente diferente:** Cuando pasamos de ImageNet (fotos del mundo natural en RGB, donde los bordes, colores y texturas son estandarizados) a imágenes con un dominio visual diametralmente ajeno (ej. ecografías médicas oscuras y ruidosas, radares espectrales, sismogramas o imágenes microscópicas especializadas). Los filtros previos no tienen correlación útil.
> 2. **Dataset inmenso a nuestra disposición:** Si disponemos de un volumen inmenso de imágenes propias (millones de recortes) para una labor ultra-específica, un modelo inicializado aleatoriamente y entrenado desde cero podrá ajustarse a nuestra tarea de nicho perfectamente sin verse "sesgado" por los pesos anteriores.

**P2. ¿Qué transforms de data augmentation añadirías al pipeline de entrenamiento para mejorar generalización?**
> **Respuesta:**
> - En la ejecución usamos aumentaciones espaciales y de color: `RandomHorizontalFlip()` (porque un gato volteado horizontalmente sigue siendo biológicamente un gato), `RandomRotation()` (para invariancia topológica si la foto de la mascota fue tomada torcida) y `ColorJitter()` (para generalizar frente a variados entornos de iluminación, flash o contrastes en interiores y exteriores). Esto fuerza a la red a no fijarse en detalles rígidos.

**P3. Investiga StepLR y CosineAnnealingLR.**
> **Respuesta:**
> - `StepLR` baja discreta y escalonadamente el Learning Rate pasadas N épocas multiplicándolo por un factor gamma (ej. dividirlo entre 10 cada 5 épocas).
> - `CosineAnnealingLR` (implementado en el experimento del respectivo notebook) reduce la curva del Learning Rate simulando el ciclo de una onda Coseno descendiente en vez de un quiebre en escalón. Esto resulta ser útil en Fine-Tuning de la capa final debido a que un descenso contínuo y suave previene inestabilidades bruscas de un salto abrupto asegurando un mínimo error de acercamiento hacia el hiperplano óptimo.

**P4. ¿Por qué ResNet usa BatchNorm y VGG no? ¿Qué efecto tiene en el entrenamiento?**
> **Respuesta:**
> - **Razón histórica:** VGG se introdujo a finales de 2014, antes del paper seminal *Batch Normalization* de Ioffe & Szegedy (2015). Su concepción precedió el descubrimiento tecnológico de dicha técnica.
> - **Efecto:** `BatchNorm` normaliza forzadamente la media a ~0 y la desviación a ~1 para las salidas ("activaciones") internas en un minilote. Esto garantiza que la red no deba lidiar continuamente con una alteración constante en las distribuciones de datos (Covariate Shift). En la práctica, esto provee una asombrosa estabilidad en entrenamiento a ResNet-18, toleran mayores *Learning Rates*, tienen regularización inherente y convergen en tan solo una fracción de las épocas requeridas por VGG-16.

**P5. Reto opcional: prueba ResNet-50 o EfficientNet-B0 y compara.**
> **Respuesta:**  
> (Teórico/Reflexivo): EfficientNet-B0 fue optimizado mediante arquitecturas de búsqueda de redes neuronales (NAS). Cuenta con una estrategia de "compound scaling" variando anchura, profundidad, y resolución de forma armónicamente pareja, logrando accuracies estadísticamente y rigurosamente superiores a las del rudimentario modelo VGG, todo esto usando tan solo un modesto cómputo (con tan solo 5.3 Millones de parámetros frente a los 11.7M de Resnet-18 y los monstruosos 138M de VGG-16).
