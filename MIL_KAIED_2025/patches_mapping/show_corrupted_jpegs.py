import os
import re

file = "/MIL_KAIED_2025/patches_mapping/errors tcga org"

with open(file, "rb") as f:
    text = f.read()

text_str = text.decode('utf-8', errors='ignore')
matches = re.findall(r'\bTCGA\S*', text_str)
unique_matches_1 = list(set(matches))
print(len(unique_matches_1))


file = "/MIL_KAIED_2025/patches_mapping/errors_tcga_norm"

with open(file, "rb") as f:
    text = f.read()

text_str = text.decode('utf-8', errors='ignore')
matches = re.findall(r'\bTCGA\S*', text_str)
unique_matches = list(set(matches))
print(len(unique_matches))



file = "/MIL_KAIED_2025/patches_mapping/corrupted_grandqc"

with open(file, "rb") as f:
    text = f.read()

text_str = text.decode('utf-8', errors='ignore')
matches = re.findall(r'\bTCGA\S*', text_str)
unique_matches = list(set(matches))
print(len(unique_matches))

wspolne = list(set(unique_matches_1) - set(unique_matches))
print(wspolne)


#['TCGA-BG-A0MI-01Z-00-DX1.5721FFD0-84E1-4C0D-9961-BEC05EDD18B9', 'TCGA-D1-A16B-01Z-00-DX1.FE32B609-4B7A-42E7-8D11-BDF0A06963DD', 'TCGA-D1-A1NW-01Z-00-DX1.04B1253F-DDEF-4EF6-828F-402C0DBC75CE', 'TCGA-B5-A11Y-01Z-00-DX1.CA396120-026C-4369-A0D8-189D19E85181', 'TCGA-AX-A1CN-01Z-00-DX2.498C651A-040E-4909-B874-30D25EC6EA3A', 'TCGA-D1-A1O7-01Z-00-DX1.3BFCD204-F894-4C44-B8C4-6630D1A1171E', 'TCGA-DI-A1BY-01Z-00-DX1.0ED757BE-AE66-442C-B3A1-5428C050D044', 'TCGA-PG-A917-01Z-00-DX3.9F8BF767-4F44-4B89-B852-FB826C3E0BAE', 'TCGA-BS-A0TE-01Z-00-DX1.D1A5A94C-7A69-4778-A879-838E7B4961D3', 'TCGA-D1-A101-01Z-00-DX1.63EFFB6C-C7F7-439D-93E6-9A798CDA6B6B', 'TCGA-B5-A1MW-01Z-00-DX1.b25dd862-1a98-472e-a4d6-53c622e8b5fb', 'TCGA-D1-A0ZO-01Z-00-DX1.05A1E6DA-52F8-460E-B675-18C850874ECC', 'TCGA-BS-A0UV-01Z-00-DX1.6C4E2500-1689-451B-BFBF-DB599177F9CB', 'TCGA-B5-A1MS-01Z-00-DX1.CB41B567-F820-45FA-8808-21BC87BAE519']

