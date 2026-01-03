def identificar_bak(ruta_archivo):
    with open(ruta_archivo, 'rb') as f:
        # Leemos los primeros 16 bytes (el encabezado)
        header = f.read(16)
        
    print(f"Header (Hex): {header.hex()}")
    print(f"Header (ASCII): {header}")

    if header.startswith(b'PK'):
        return "Es un ZIP. Renómbralo a .zip"
    elif header.startswith(b'SQLite format 3'):
        return "Es una base de datos SQLite. Renómbralo a .db"
    else:
        return "Formato desconocido o encriptado."

# Solo cambia la ruta por la de tu backup de ReadEra
print(identificar_bak("ReadEra-Premium_2026-01-02_18.22.bak"))