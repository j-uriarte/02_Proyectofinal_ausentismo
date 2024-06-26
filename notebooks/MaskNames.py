import pandas as pd

def main():
    # Cargar los datos desde un archivo CSV
    file_path = "../data/Compiled ABS 2023. V2.csv"
    try:
        datos = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"El archivo no se encontró en la ruta especificada: {file_path}")
        return

    # Extraer los nombres únicos para cada columna que deseamos reemplazar
    unique_names = pd.unique(datos['Name'])
    unique_coaches = pd.unique(datos['Coach'])
    unique_oms = pd.unique(datos['OM'])

    # Crear mapas de reemplazo para cada columna
    agent_map = {name: f"Agente {i+1}" for i, name in enumerate(unique_names)}
    coach_map = {name: f"Coach {i+1}" for i, name in enumerate(unique_coaches)}
    om_map = {name: f"OM {i+1}" for i, name in enumerate(unique_oms)}

    # Aplicar los mapas de reemplazo a las columnas correspondientes
    datos['Name'] = datos['Name'].replace(agent_map)
    datos['Coach'] = datos['Coach'].replace(coach_map)
    datos['OM'] = datos['OM'].replace(om_map)

    # Guardar los datos modificados de vuelta a un archivo CSV
    output_file_path = "../data/Compiled_ABS_2023_V2_MaskedNames.csv"
    datos.to_csv(output_file_path, index=False)
    print(f"Datos actualizados guardados en {output_file_path}")

if __name__ == "__main__":
    main()

