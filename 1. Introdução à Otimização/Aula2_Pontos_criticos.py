#1. Otimização Univariada: Manufatura de MóveisNeste exemplo, usamos a função de lucro $P(q) = 40q - 0.5q^2$. 
#Vamos calcular a primeira derivada para encontrar o ponto crítico e a segunda derivada para provar que é um máximo.  
import sympy as sp

# 1. Definindo a variável simbólica
q = sp.symbols('q')

# 2. Definindo a função objetivo (Lucro)
P = 40*q - 0.5*q**2
print(f"Função Objetivo P(q) = {P}\n")

# 3. Condição Necessária de Primeira Ordem (Primeira Derivada)
dP_dq = sp.diff(P, q)
print(f"Primeira derivada (dP/dq): {dP_dq}")

# Encontrando o ponto crítico (igualando a derivada a zero)
q_opt = sp.solve(dP_dq, q)
print(f"Ponto crítico encontrado (q*): {q_opt[0]}\n")

# 4. Condição Suficiente de Segunda Ordem (Segunda Derivada)
d2P_dq2 = sp.diff(dP_dq, q) # Derivada da derivada
print(f"Segunda derivada (d²P/dq²): {d2P_dq2}")

# Classificando o ponto
valor_segunda_derivada = d2P_dq2.subs(q, q_opt[0])

if valor_segunda_derivada < 0:
    print("Conclusão: Como a segunda derivada é negativa (< 0), confirmamos um Ponto de MÁXIMO!")
    lucro_maximo = P.subs(q, q_opt[0])
    print(f"Lucro Máximo atingido: ${lucro_maximo}")
elif valor_segunda_derivada > 0:
    print("Conclusão: Como a segunda derivada é positiva (> 0), temos um Ponto de MÍNIMO.")
else:
    print("Conclusão: O teste da segunda derivada é inconclusivo.")

#2. Otimização Multivariada: Fábrica de BiscoitosAqui lidamos com a função de produção Cobb-Douglas $F(K,L) = K^{0.34} L^{0.66}$.
  # Em vez de uma única derivada, calculamos o vetor gradiente (as derivadas parciais) para buscar pontos críticos. 
import sympy as sp

# 1. Definindo as variáveis simbólicas
K, L = sp.symbols('K L')

# 2. Definindo a função objetivo (Produção)
F = K**0.34 * L**0.66
print(f"Função de Produção F(K,L) = {F}\n")

# 3. Calculando o Vetor Gradiente (Derivadas Parciais)
dF_dK = sp.diff(F, K)
dF_dL = sp.diff(F, L)

print("Vetor Gradiente:")
print(f"∂F/∂K: {dF_dK}")
print(f"∂F/∂L: {dF_dL}\n")

# 4. Buscando Pontos Críticos
# Tentamos resolver o sistema onde ambas as derivadas parciais são zero simultaneamente
pontos_criticos = sp.solve([dF_dK, dF_dL], (K, L))

print(f"Pontos críticos encontrados: {pontos_criticos}")

if not pontos_criticos:
    print("Conclusão: A lista está vazia. Não existem máximos ou mínimos finitos para esta função.")
    print("A produção cresce indefinidamente à medida que K e L aumentam.")
