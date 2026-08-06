---
name: calcular
description: Resolver una operacion o expresion aritmetica (por ejemplo "12 * (3 + 4)").
handler: eval_aritmetico
---
Esta skill resuelve expresiones aritmeticas de forma DETERMINISTA, con un
evaluador seguro en Python (no usa el LLM, que suele equivocarse en cuentas).

El campo `handler: eval_aritmetico` del header le indica al programa que, en vez
de mandar un prompt al modelo, ejecute la funcion Python registrada con ese
nombre. Sirve para mostrar que una skill puede estar respaldada por codigo y no
solo por instrucciones en lenguaje natural.
