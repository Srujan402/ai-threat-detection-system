
from django import forms


class CSVUploadForm(forms.Form):
    DATA_TYPES = [
        ("network", "Network Flow CSV"),
        ("system", "Security Log CSV"),
    ]

    data_type = forms.ChoiceField(choices=DATA_TYPES)
    csv_file = forms.FileField()

    def clean_csv_file(self):
        uploaded_file = self.cleaned_data["csv_file"]

        if not uploaded_file.name.lower().endswith(".csv"):
            raise forms.ValidationError("Please upload a .csv file.")

        if uploaded_file.size > 20 * 1024 * 1024:
            raise forms.ValidationError("The file must be 20 MB or smaller.")

        return uploaded_file
