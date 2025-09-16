from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('app', '0002_add_fields_and_jobs'),
    ]

    operations = [
        migrations.CreateModel(
            name='ScrapeSource',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('url', models.URLField(max_length=1024, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={'ordering': ['name']},
        ),
    ]

