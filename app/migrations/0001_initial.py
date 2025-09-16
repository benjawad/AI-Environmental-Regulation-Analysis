from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=512)),
                ('doc_type', models.CharField(blank=True, default='', max_length=128)),
                ('pages_count', models.PositiveIntegerField(default=0)),
                ('avg_confidence', models.FloatField(default=0.0)),
                ('json_path', models.CharField(blank=True, default='', max_length=1024)),
                ('source_pdf', models.CharField(blank=True, default='', max_length=1024)),
                ('status', models.CharField(blank=True, default='', max_length=32)),
                ('error', models.TextField(blank=True, default='')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={'ordering': ['-created_at']},
        ),
        migrations.CreateModel(
            name='AnalysisResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.TextField(blank=True, default='')),
                ('structured_data', models.JSONField(blank=True, default=list)),
                ('rows_count', models.PositiveIntegerField(default=0)),
                ('status', models.CharField(blank=True, default='', max_length=32)),
                ('error', models.TextField(blank=True, default='')),
                ('pdf_path', models.CharField(blank=True, default='', max_length=1024)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={'ordering': ['-created_at']},
        ),
    ]

