# Heftromane


Dies ist ein Repositorium, das Programmcode zur Analyse von Heftromanen enthält. 
Im Zentrum steht die Operationalisierung und Analyse von Spannung (suspense).

Warnung: Das Repositorium befindet sich gegenwärtig im Aufbau. Dokumentationen sind daher noch unvollständig und für das Funktionieren der Module gibt es keine Gewähr. Klassen, Methoden und Funktionen sind zum großen Teil noch nicht erläutert.

Die Skripte zur Durchführung der Analysen befinden sich im Ordner "scripts_Heftromane". Die Module  "semantic_analysis" und "sent_analysis" enthalten die Klassen und Funktionen zur Analyse der plotbasierten und emotionsbasierten Dimensionen von Spannung. 

Die Klassen "Text" im Modul preprocessing.text sowie "DocumentFeatureMatrix" in preprocessing.corpus sind die Basisklassen zur Verarbeitung der Texte auf Text-, sowie Korpusebene. U.a. die Klassen "FearShare" und DocSentFearMatrix im Modul sent_analysis.fear sowie SettingShare und DocThemesMatrix im Modul semantic_analysis.themes  sind von diesen abgeleitet.
