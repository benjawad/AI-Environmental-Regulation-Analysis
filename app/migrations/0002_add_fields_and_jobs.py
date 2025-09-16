from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='json_content',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='analysisresult',
            name='raw_result',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='analysisresult',
            name='kb_snapshot',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.CreateModel(
            name='ScrapeJob',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('base_url', models.CharField(max_length=1024)),
                ('params', models.JSONField(blank=True, null=True)),
                ('status', models.CharField(blank=True, default='pending', max_length=32)),
                ('error', models.TextField(blank=True, default='')),
                ('stats', models.JSONField(blank=True, null=True)),
                ('links', models.JSONField(blank=True, null=True)),
                ('files', models.JSONField(blank=True, null=True)),
                ('output_dir', models.CharField(blank=True, default='', max_length=1024)),
                ('started_at', models.DateTimeField(auto_now_add=True)),
                ('finished_at', models.DateTimeField(blank=True, null=True)),
            ],
            options={'ordering': ['-started_at']},
        ),
        migrations.CreateModel(
            name='GeneratedRegister',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('kind', models.CharField(choices=[('legal', 'Legal Register'), ('commitment', 'Commitment Register')], max_length=32)),
                ('file_path', models.CharField(max_length=1024)),
                ('file_size', models.BigIntegerField(default=0)),
                ('sha256', models.CharField(blank=True, default='', max_length=64)),
                ('rows_count', models.PositiveIntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('analysis', models.ForeignKey(blank=True, null=True, on_delete=models.deletion.SET_NULL, to='app.analysisresult')),
            ],
            options={'ordering': ['-created_at']},
        ),
    ]

