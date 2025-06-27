from django import template

register = template.Library()
print("custom_tags is being loaded!")


@register.filter
def split_tags(value):
    """Ensure it works if it's a list of tags."""
    if isinstance(value, str):
        l=[]
        for tag in value.split(','):
            if tag.startswith('['):   #for ["xyz",..
                tag=tag[2:-1]
                tag.strip()

            elif tag.endswith(']'):  #for ..,"xyz"]
                tag=tag[2:-2]
                tag.strip()

            else:
                tag=tag[1:-1]   # for "xyz"
                tag.strip()
            l.append(tag)
        return l
    return []
