{{- define "atlas.labels" -}}
app.kubernetes.io/part-of: atlas
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
