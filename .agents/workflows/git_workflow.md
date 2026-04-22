---
description: Metodología estricta de Git y gestión de Historias de Usuario (HU).
---
# Flujo de Trabajo Estricto de Github e Historias de Usuario

Este workflow **DEBE** seguirse rigurosamente por el agente para garantizar una limpieza total en el control de versiones y trazabilidad de las *Historias de Usuario (HU)* establecidas en `doc/historias_de_usuario.md`. Bajo ninguna circunstancia se deben agrupar múltiples HU resueltas en una misma Pull Request si no han sido explícitamente autorizadas por el usuario.

## Regla de Oro
**1 Historia de Usuario (HU) = 1 Rama (Branch) = 1 Pull Request (PR)**

## Pasos Estrictos de Ejecución:

1. **Lectura de la Historia:** Antes de comenzar a programar, lee la HU designada a resolver en el backlog.
2. **Creación de Rama Aislada:** Desde la terminal, sitúate en la rama base (por defecto `main` o la que el usuario indique) y crea una rama exclusiva para esta historia con el formato `feature/hu-XXX-nombre-descriptivo`.
   `git checkout main && git pull`
   `git checkout -b feature/hu-XXX-descripcion`
3. **Implementación:** Programa el código exacto necesario para satisfacer **ÚNICAMENTE** los Criterios de Aceptación estipulados para esa HU en particular.
4. **Validación:** Comprueba y marca la historia como `[x]` en el backlog si has completado sus subtareas (y haz un `git add` sobre el backlog editado + el código).
5. **Generar Pull Request Independiente:** 
   `git commit -m "feat(HU-XXX): Descripción corta de la resolución"`
   `git push -u origin feature/hu-XXX-descripcion`
   Posteriormente, usa la CLI de Github para abrir la PR de manera autónoma:
   `gh pr create --title "feat: Resolución de HU-XXX - Título" --body "..."`
6. **Espera de Aprobación:** Nunca cierres la siguiente HU sin que esta PR haya sido validada/mergeada o sin indicación expresa del usuario.
