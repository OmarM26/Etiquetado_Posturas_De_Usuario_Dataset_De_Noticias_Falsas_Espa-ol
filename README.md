# Etiquetado de Posturas de Usuarios en Dataset de Noticias Falsas en Español

Este repositorio contiene el código, configuraciones y recursos asociados al proyecto **Etiquetado de Posturas de Usuarios en Dataset de Noticias Falsas en Español**, desarrollado como Trabajo de Título para optar al grado de **Ingeniero Civil en Informática** en la **Universidad de Valparaíso**.

El objetivo del proyecto es generar un proceso de **clasificación automática de posturas de usuarios en conversaciones de Twitter (X)**, permitiendo etiquetar las respuestas de los usuarios frente a una publicación informativa en distintas categorías de postura.

---

## Descripción

El crecimiento de las redes sociales ha facilitado la rápida difusión de información, pero también ha incrementado la propagación de rumores y noticias falsas. En este contexto, los propios usuarios juegan un rol clave a través de sus interacciones, ya sea apoyando, rechazando, comentando o cuestionando una publicación.

Este trabajo aborda la tarea de **detección de postura (stance classification)** en idioma español, un área donde aún existen limitados recursos y estudios en comparación con el inglés. Para ello, se desarrollan y evalúan modelos de aprendizaje automático y aprendizaje profundo aplicados a conversaciones reales extraídas de Twitter.

Las posturas clasificadas corresponden a:

- De acuerdo  
- Desacuerdo  
- Consulta  
- Comenta  

Cada respuesta es etiquetada considerando su relación con el mensaje raíz de la conversación.

---

## Enfoque del Proyecto

El repositorio implementa dos enfoques de clasificación de postura:

### 1. Modelo entrenado en inglés y aplicado a español
Se entrena un modelo basado en una arquitectura **BI-LSTM** utilizando un dataset en inglés (RumourEval). Posteriormente, el modelo es utilizado para etiquetar conversaciones en español mediante transferencia de aprendizaje.

### 2. Modelo entrenado directamente en español
Se utiliza un modelo **BERT preentrenado en español (BETO)**, el cual es ajustado utilizando datos del dataset CLNews etiquetados manualmente para la tarea de postura.

Ambos enfoques son comparados para analizar su desempeño en el contexto del idioma español.

---

## Dataset

El proyecto trabaja con el dataset **CLNews**, desarrollado por la Universidad de Valparaíso, el cual contiene conversaciones en Twitter asociadas a noticias verificadas y no verificadas sobre distintos eventos ocurridos en Chile.

Este trabajo incorpora etiquetas de postura de usuarios, complementando las etiquetas de veracidad de las noticias ya existentes en el dataset.

---

## Requerimientos

El proyecto ha sido desarrollado utilizando:

- Python 3.x  
- PyTorch  
- Transformers (HuggingFace)  
- Numpy  
- Pandas  
- Scikit-learn  

Las dependencias específicas pueden encontrarse en el archivo `requirements.txt`.

---


## Resultados

Los modelos son evaluados utilizando métricas estándar de clasificación, tales como precisión, recall y F1-score, además de matrices de confusión y análisis de errores en hilos de conversación específicos.

Los resultados permiten comparar el rendimiento entre un enfoque de transferencia desde inglés y un enfoque entrenado directamente en español.

---

## Contexto Académico

Este repositorio forma parte del Trabajo de Título presentado en la **Escuela de Ingeniería Informática de la Universidad de Valparaíso**, y tiene como finalidad aportar al estudio de la detección de postura de usuarios en redes sociales en idioma español.

---

## Autor

Omar Martínez Tobar  
Escuela de Ingeniería Civil Informática  
Universidad de Valparaíso  


*nota: este repositorio no tiene traducción al inglés ya que está pensado a usuarios del habla hispana.*

*note: this repositoy no have an english version because this thinked to spanish users.*

